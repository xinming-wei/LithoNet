import os
import torch
import numpy as np
from options.test_options import TestOptions
from data.dataloader import create_dataset
from models import create_model
from util.visualizer import Visualizer
import functools
import cv2
import matplotlib.pyplot as plt


def compute_epe(orig_layout, fake_image, aerial_thresh):
    """input:
       real_layout: np.load(opt.layout_path)[index]
       fake_image:  model.fake_aerial_image.cpu()
    """
    # Compare the location of two rectangles
    # Note that margin is reserved when comparing
    def cmp_rect(rect1, rect2):
        if rect1[0] < rect2[0] - 60:
            return -1
        elif rect1[0] <= rect2[0] + 60 :
            if rect1[1] < rect2[1] - 60:
                return -1
            elif rect1[1] <= rect2[1] + 60:
                return 0
            else:  
                return 1
        else:
            return 1
    
    # upsample to 4000*4000
    orig = cv2.resize(orig_layout, (4000, 4000), interpolation=cv2.INTER_LINEAR)
    fake = cv2.resize(fake_image, (4000, 4000), interpolation=cv2.INTER_LINEAR)
    
    # thresholding
    _, orig = cv2.threshold(orig, 0.50, 255, cv2.THRESH_BINARY)
    _, fake = cv2.threshold(fake, aerial_thresh, 255, cv2.THRESH_BINARY_INV)

    titles = ['original layout', 'fake layout']
    images = [orig, fake]
    rects = {'original layout':[], 'fake layout':[]}

    # Find all contours and draw bounding rectangles
    for i, image in enumerate(images):
        # print(titles[i] + ' coordinates:')
        image = np.array(image, dtype=np.uint8)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for j, cntr in enumerate(contours):
            x,y,w,h = cv2.boundingRect(cntr)
            # print("contact %d: (%d, %d), w = %d, h = %d" % (j, x, y, w, h))
            if i > 0 and w * h < 0.2 * 120 * 120:   # remove noise in golden and fake layout
                continue
            rects[titles[i]].append((x, y, x+w, y+h)) 
        rects[titles[i]].sort(key=functools.cmp_to_key(cmp_rect))
    
    epes = []
    fail = False
    try:
        for i, rect in enumerate(rects['original layout']):
            epes.append(0.5 * max([abs(rect[k] - rects['fake layout'][i][k]) for k in range(0,4)]))
    except:  # Fail to compute EPE because noise rectangles
        fail = True
    return epes, fail

def epe_hist(epes, path):
    plt.figure()
    bins = np.arange(0.0, 20.0, 0.5)
    plt.hist(epes, bins=bins)
    plt.xlabel('EPE (nm)')
    plt.xticks(np.arange(0, 20, 1.0))
    plt.ylabel('Number')
    plt.title('Distribution of EPEs')
    plt.savefig(path)
    plt.show()

def plot(real_layout, fake_mask, fake_aerial_image, fake_layout, path):
    titles = ['real layout', 'fake mask', 'fake layout', 'fake aerial image']
    images = [real_layout, fake_mask, fake_layout, fake_aerial_image]

    plt.figure()
    for i, image in enumerate(images):
        plt.subplot(2,2,i+1)
        plt.title(titles[i])
        plt.imshow(image)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def mse(nparray1, nparray2):
    """calculate the Mean Squared Error between two nparrays"""
    width = nparray1.shape[1]
    return float((np.linalg.norm(nparray1 - nparray2))**2 / width**2)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.name = 'ilt_finetune'
    opt.phase = 'finetune'
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
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    results_path = os.path.join(opt.checkpoints_dir, opt.name, "test_results")
    ls_squared_l2, epe_total = [], []

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        index = int(data['index'])

        # Remove the paddings of images and transfer them to CPU numpy arrays
        img_width = opt.load_size
        pad_width = dataset.dataset.pad_width
        real_sliced_layout = model.real_sliced_layout[0, 0, pad_width:img_width-pad_width, pad_width:img_width-pad_width].cpu().numpy()
        fake_mask = model.fake_mask[0, 0, pad_width:img_width-pad_width, pad_width:img_width-pad_width].cpu().numpy()
        _, fake_mask = cv2.threshold(fake_mask, 0, 255, cv2.THRESH_BINARY)
        fake_aerial_image = model.fake_aerial_image.relu()[0, 0, pad_width:img_width-pad_width, pad_width:img_width-pad_width]
        fake_sliced_layout = torch.heaviside(-(fake_aerial_image - model.aerial_thresh), values=torch.tensor([1.0]).to(model.device)).cpu().numpy()
        fake_aerial_image = fake_aerial_image.cpu().numpy()

        # Compute squared_l2 error
        squared_l2 = mse(real_sliced_layout, fake_sliced_layout)
        ls_squared_l2.append(squared_l2)
        # Compute EPEs
        epes, epe_fail = compute_epe(real_sliced_layout, fake_aerial_image, model.aerial_thresh)
        for k in epes:
            if k < 20: epe_total.append(k)

        visualizer.print_test_results(index, squared_l2, epes, epe_fail)
        # Plot results images and save them to the disk
        if opt.plot:
            os.system("mkdir -p %s" % (results_path + '/test_images/'))
            image_path = os.path.join(results_path, 'test_images', '%04d.jpg' % i)
            print("########## Plotting (%04d)-th image ###########" % i)
            plot(real_sliced_layout, fake_mask, fake_aerial_image, fake_sliced_layout, path=image_path)
            print("########## Finish plotting ###########")

    visualizer.print_l2_metrics(ls_squared_l2)
    # Plot EPE distribution and save them into disk
    epe_hist(epe_total, path=os.path.join(results_path, 'EPE_hist'))
