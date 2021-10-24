import torch
from .base_model import BaseModel
from . import networks
import os


class DiffHeaviside(torch.autograd.Function):
    """
    A customized differentiable Heaviside function, forward with Heaviside and backprop with sigmoid
    -- to instantiate, use '... = DiffHeaviside.apply'
    """
    @staticmethod
    def forward(ctx, input, slope=60):
        # Typical Heaviside function 
        #'slope' determines the steepness of the sigmoid approximating heaviside
        ctx.save_for_backward(input)
        ctx.slope = slope
        tensor = input.clone()
        tensor[tensor <= 0.0] = 0.0
        tensor[tensor > 0.0] = 1.0
        return tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        # return as many gradients as there are input arguments
        input, = ctx.saved_tensors
        c = ctx.slope
        return grad_output * c*torch.exp(-c * input) / (1 + torch.exp(-c * input))**2, None


class PretrainModel(BaseModel):
    """ This class implements the pretraining of ILT model with pix2pix model
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.set_defaults(n_epochs=25, n_epochs_decay=25)
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        """Initialize the PretrainModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1', 'G_GAN', 'D_real', 'D_fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        # input['layout'].shape: (batch_size, 1, 256, 256)
        # After data preprocessing: {-1, 1} => {other region, contact region}
        self.real_layout = input['layout'].to(self.device, dtype=torch.float)
        self.real_mask = input['mask'].to(self.device, dtype=torch.float)
    
    def sigmoid(self, tensor, slope):
        """ Approximate heaviside with sigmoid since 
        no derivative is implemented for heaviside in pytorch"""
        return 1 / (1 + torch.exp( - slope * tensor))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        G_output = self.netG(self.real_layout)  # G(A)
        ## Don't binarize when pretrain
        self.fake_mask =  G_output

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_layout, self.fake_mask), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_layout, self.real_mask), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_layout, self.fake_mask), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_mask, self.real_mask) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights


class FinetuneModel(BaseModel):
    """ This class implements the finetuning model

    Here, we train the Generator in an auto-encoder manner (i.e. without GAN loss)
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.set_defaults(n_epochs=15, n_epochs_decay=15, save_epoch_freq=10)
        # lower lr when finetuning
        parser.set_defaults(lr=0.00004)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L2', type=float, default=1000.0, help='weight for L2 loss')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['Squared_L2', 'G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.loss_names = ['G_L2']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        # define the pretrained generator to be finetuned
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # define the pretrained litho simulation generator network
        self.netMask2Image = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # Threshold we use to slice aerial images into layouts (wafer image)
        self.aerial_thresh = opt.aerial_thresh
          
        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss() # Use L2 loss instead of L1
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def setup(self, opt):
        """Overwrite the setup function in Basemodel to adjust finetune features"""
        backbone_path = os.path.join(opt.checkpoints_dir, 'ilt_pretrain', 'latest_net_G.pth')
        mask2image_path = os.path.join(opt.checkpoints_dir, 'litho_simul', 'latest_net_G.pth')
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            self.load_pretrained_net(backbone_path, 'netG')
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.load_pretrained_net(mask2image_path, 'netMask2Image')
        self.print_networks(opt.verbose)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # input['layout'].shape: (batch_size, 1, 256, 256)
        # After data preprocessing: {-1, 1} => {other region, contact region}
        self.real_layout = input['layout'].to(self.device, dtype=torch.float)
    
    def sigmoid(self, tensor, slope):
        """ Approximate heaviside with sigmoid since 
        no derivative is implemented for heaviside in pytorch"""
        return 1 / (1 + torch.exp( - slope * tensor))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        G_output = self.netG(self.real_layout)  # G(A)
        # Binarize the output mask image with Heavside in forward-prop
        Heaviside = DiffHeaviside.apply
        self.fake_mask = 2 * Heaviside(G_output) - 1

        self.set_requires_grad(self.netMask2Image, False)
        self.fake_aerial_image = self.netMask2Image(self.fake_mask)
        self.fake_sliced_layout = 1 - self.sigmoid(self.fake_aerial_image - self.aerial_thresh, slope=400)
        self.real_sliced_layout = torch.heaviside(self.real_layout - 0, values=torch.tensor([1.0]).to(self.device))

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_L2 = self.criterionL2(self.fake_sliced_layout, self.real_sliced_layout) * self.opt.lambda_L2
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L2
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
