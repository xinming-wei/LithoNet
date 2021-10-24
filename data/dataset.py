import torch
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import random
from abc import abstractmethod
import os


class BaseDataset(data.Dataset):

    def __init__(self, opt):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.pad_width = 0
    
    @abstractmethod
    def __len__(self):
        return 0
    
    @abstractmethod
    def __getitem__(self):
        pass

    def name(self):
        return 'BaseDataset'


def npz_path(path):
    file = os.listdir(path)
    assert len(file) == 1, "There should be one and only one .npz dataset file be put in %s" % path
    filename = file[0]
    _, type = os.path.splitext(filename)
    assert type == '.npz', "Dataset should be packed as .npz files"
    res_path = os.path.join(path, filename)
    return res_path


class LithoSimulDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.mask_root = npz_path(os.path.join(opt.dataroot, 'litho_simul/train/mask')) if opt.isTrain else npz_path(os.path.join(opt.dataroot, 'litho_simul/test/mask'))
        self.aerial_image_root = npz_path(os.path.join(opt.dataroot, 'litho_simul/train/aerial_image')) if opt.isTrain else npz_path(os.path.join(opt.dataroot, 'litho_simul/test/aerial_image'))
    
    def __getitem__(self, index):
        mask = np.load(self.mask_root)["{:04}".format(index)]
        aerial_image = np.load(self.aerial_image_root)["{:04}".format(index)]
        mask_tensor, aerial_image_tensor = self.preprocess(mask, aerial_image)

        input_dict = {'mask':mask_tensor, 'aerial_image':aerial_image_tensor, 'index':index}
        return input_dict
    
    def __len__(self):
        len1, len2 = len(np.load(self.mask_root)), len(np.load(self.aerial_image_root))
        assert len1 == len2, "To make pairs, two corresponding datasets should have the same size"
        return len1
    
    def preprocess(self, mask, aerial_image):
        orig_size = mask.shape[1]
        load_size = self.opt.load_size  # 2^n
        assert mask.shape == aerial_image.shape and mask.ndim == aerial_image.ndim == 2, "layout and mask should have the same shape"
        assert load_size >= orig_size, "Image load size should be no less than original size "
        # Add zero padding to make image width 2^n
        self.pad_width = (load_size - orig_size) // 2
        mask, aerial_image = np.pad(mask, self.pad_width, 'constant'), np.pad(aerial_image, self.pad_width, 'constant')

        transform_mask = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        # values of aerial images should be maintained and not be normalized
        aerial_image_tensor = torch.from_numpy(aerial_image).float()
        aerial_image_tensor.unsqueeze_(0)
        return transform_mask(mask), aerial_image_tensor

    def name(self):
        return 'LithoSimulDataset'


class PretrainDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.mask_root = npz_path(os.path.join(opt.dataroot, 'ilt_pretrain/mask')) 
        self.layout_root = npz_path(os.path.join(opt.dataroot, 'ilt_pretrain/layout')) 
    
    def __getitem__(self, index):
        # The .npz file should be indexed like ['0003'], ['0012']
        layout = np.load(self.layout_root)["{:04}".format(index)]
        mask = np.load(self.mask_root)["{:04}".format(index)]
        layout = np.array(layout, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)

        layout_tensor, mask_tensor = self.preprocess(layout, mask)
        input_dict = {'layout':layout_tensor, 'mask':mask_tensor, 'index':index}
        return input_dict

    def __len__(self):
        len1, len2 = len(np.load(self.layout_root)), len(np.load(self.mask_root))
        assert len1 == len2, "To make pairs, two corresponding datasets should have the same size"
        return len1
    
    def preprocess(self, layout, mask):
        orig_size = layout.shape[1]
        load_size = self.opt.load_size
        assert layout.shape == mask.shape and layout.ndim == mask.ndim == 2, "layout and mask should have the same shape"
        assert load_size >= orig_size, "Image load size should be no less than original size "
        self.pad_width = (load_size - orig_size) // 2
        layout, mask = np.pad(layout, self.pad_width, 'constant'), np.pad(mask, self.pad_width, 'constant')

        # data augmentation
        if random.random() > 0.5:
            if random.random() > 0.5:
                # Up-down flip
                layout = np.flip(layout, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
            else:
                # Left-right flip
                layout = np.flip(layout, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
        
        # Pixel values range in [-1, 1] 
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        return transform(layout), transform(mask)

    def name(self):
        return "PretrainDataset"

    
class FinetuneDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.layout_root = npz_path(os.path.join(opt.dataroot, 'ilt_finetune/train')) if opt.isTrain else npz_path(os.path.join(opt.dataroot, 'ilt_finetune/test'))

    def __getitem__(self, index):
        layout = np.load(self.layout_root)["{:04}".format(index)]
        layout = np.array(layout, dtype=np.uint8)
        layout_tensor = self.preprocess(layout)

        input_dict = {'layout':layout_tensor, 'index':index}
        return input_dict
    
    def __len__(self):
        return len(np.load(self.layout_root))

    def preprocess(self, layout):
        orig_size = layout.shape[1]
        load_size = self.opt.load_size
        assert load_size >= orig_size, "Image load_size should be larger than original size"
        self.pad_width = (load_size - orig_size) // 2
        layout = np.pad(layout, self.pad_width, 'constant')

        if self.opt.isTrain:
            # only do data-augmentation when training
            if random.random() > 0.5:
                if random.random() > 0.5:
                    # Up-down flip
                    layout = np.flip(layout, axis=0).copy()
                else:
                    # Left-right flip
                    layout = np.flip(layout, axis=1).copy()
        
        # Pixel values range in [-1, 1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        
        return transform(layout)
