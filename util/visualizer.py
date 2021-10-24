import numpy as np
import os
import time
from . import util


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.
    """
    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.name = opt.name
        
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)
        else: # save testing results
            os.system("mkdir -p %s" % os.path.join(opt.checkpoints_dir, opt.name, "test_results"))
            self.test_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'test_results', 'test_log.txt')
            with open(self.test_log_name, 'a') as log_file:
                now = time.strftime("%c")
                log_file.write('================ Test Results (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
    
    def print_test_results(self, index, l2_error, epes, epe_fail):
        """print current EPEs of each contact per sample on console & the disk
        Parameters:
            index (int) -- # of test sample
            metrics (tuple) -- (rmse, nrmse) per sample
            epes (tuple) -- (epes_golden(list), epes_fake(list))
            epe_fail (bool) -- if epe is sunccessfully calculated
        """
        def print_and_save(message):  # print and save the message
            print(message)
            with open(self.test_log_name, "a") as log_file:
                log_file.write('%s\n' % message)
        
        message = "************ sample [%04d] ************" % index
        print_and_save(message)    
        message = "Mean squared L2 error(%%): %.5f" % (100.0 * l2_error)
        print_and_save(message)
        if epe_fail:
            message = "EPE calculation failed: check the sample images!"
            print_and_save(message)
            return

        message = "EPEs of fake layout (unit: nm):"
        print_and_save(message)
        for i, epe in enumerate(epes):
            message = "contact %d: %.1f" % (i+1, epe)
            print_and_save(message)
        
    def print_l2_metrics(self, ls_squared_l2):  
        message = "================= General Test Results ================="
        print(message)
        with open(self.test_log_name, "a") as log_file:
            log_file.write('%s\n\n' % message)

        message = "Test Mean Squared L2 Error(%):\n" + "average: %.5f max: %.5f std: %.5f" % (100.0*np.average(ls_squared_l2), 100.0*max(ls_squared_l2), 100.0*np.std(ls_squared_l2))
        print(message)
        with open(self.test_log_name, "a") as log_file:
            log_file.write('%s\n' % message)
    
    def print_rmse_metrics(self, ls_rmse, ls_nrmse):
        message = "================= General Test Results ================="
        print(message)
        with open(self.test_log_name, "a") as log_file:
            log_file.write('%s\n\n' % message)
            
        message = "Test RMSE(xE-4):\n" + "average: %.2f max: %.2f std: %.2f\n" % (10000.0*np.average(ls_rmse), 10000.0*max(ls_rmse), 10000.0*np.std(ls_rmse))
        message += "Test NRMSE(%%):\n" + "average: %.2f max: %.2f std: %.2f" % (100.0*np.average(ls_nrmse), 100.0*max(ls_nrmse), 100.0*np.std(ls_nrmse))
        print(message)
        with open(self.test_log_name, "a") as log_file:
            log_file.write('%s\n\n' % message)


