import torch.utils.data
from data.dataset import FinetuneDataset, PretrainDataset, LithoSimulDataset

def create_dataset(opt):
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():

    def __init__(self, opt):
        self.opt = opt
        if opt.phase == 'litho_simul':
            self.dataset = LithoSimulDataset(opt)
        elif opt.phase == 'pretrain':
            self.dataset = PretrainDataset(opt)
        elif opt.phase == 'finetune':
            self.dataset = FinetuneDataset(opt)
        else:
            raise Exception("Arg 'phase' should be [litho_simul | pretrain | finetune]")
            
        print("dataset [%s] was created"% self.dataset.name())
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size = opt.batch_size,
            shuffle = not opt.serial_batches,
            num_workers = int(opt.num_threads)
        )
    
    def load_data(self):
        return self
    
    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
    
    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data