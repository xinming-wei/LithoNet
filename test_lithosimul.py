import os
import torch
import numpy as np
import random
from options.test_options import TestOptions
from data.dataloader import create_dataset
from models import create_model
from util.visualizer import Visualizer
import matplotlib.pyplot as plt


def compute_metrics(pred_img, real_img):
    """Compute RMSE and NRMSE between two images
    input: torch.tensor
    """
    l2norm_delta = float(torch.linalg.norm(pred_img - real_img))
    l2norm_real = float(torch.linalg.norm(real_img))
    rmse = l2norm_delta / pred_img.shape[1]
    nrmse = l2norm_delta / l2norm_real
    return rmse, nrmse

def plot(mask, real_aerial_image, fake_aerial_image, path):

    titles = ['mask', 'real aerial image', 'fake aerial image']
    images = [mask, real_aerial_image, fake_aerial_image]

    fig, axs = plt.subplots(1, 3)
    for i, image in enumerate(images):
        axs[i].set_title(titles[i])
        _image = axs[i].imshow(image)
        if i > 0:
            fig.colorbar(_image, ax=axs[i], shrink=0.3)

    plt.tight_layout()
    plt.savefig(path)
    plt.show()


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.name = 'litho_simul'
    opt.phase = 'litho_simul'   # Explicitly set phase here
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    if opt.load_size == None:  opt.load_size = 256

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    if opt.eval:
        model.eval()

    rmse, nrmse = [], []
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference

        img_width = opt.load_size
        pad_width = dataset.dataset.pad_width
        real_aerial_image = model.real_B[0, 0, pad_width:img_width-pad_width, pad_width:img_width-pad_width].relu()
        fake_aerial_image = model.fake_B[0, 0, pad_width:img_width-pad_width, pad_width:img_width-pad_width]
        mask = model.real_A[0, 0, pad_width:img_width-pad_width, pad_width:img_width-pad_width]

        _rmse, _nrmse = compute_metrics(fake_aerial_image.cuda(), real_aerial_image.cuda())
        rmse.append(_rmse)
        nrmse.append(_nrmse)
     
        print('processing (%04d)-th image...' % i)
        print('RMSE(xE-4):', 10000*_rmse, '  NRMSE(%):', 100*_nrmse)

        if opt.plot:
            image_path = (os.path.join(opt.checkpoints_dir, opt.name, 'test_results', 'test_images'))
            os.system("mkdir -p %s" % image_path)
            print("########## Plotting (%04d)-th image ###########" % i)
            plot(mask.cpu().numpy(), real_aerial_image.cpu().numpy(), fake_aerial_image.cpu().numpy(), path=image_path+"/{:04}.jpg".format(i))
            print("########## Finish plotting ###########")
      
    visualizer.print_rmse_metrics(rmse, nrmse)
