import glob, torch
import netCDF4 as nc
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class get_data(Dataset):
    def __init__(self, inp_path, tar_path, train):

        self.inp_path = inp_path
        self.tar_path = tar_path
        self.train = train

        # get file list
        self.inp_flist = glob.glob(self.inp_path+'*.nc')
        self.tar_flist = glob.glob(self.tar_path+'*.nc')

        # sorting
        self.inp_flist.sort()
        self.tar_flist.sort()
            
        # get data stats
        self.nyr = len(self.inp_flist)

        with nc.Dataset(self.inp_flist[0], 'r') as f:
            self.inp_zdim = f['p'].shape[1]
            self.ydim = f['p'].shape[2] - 21
            self.xdim = f['p'].shape[3]          

        with nc.Dataset(self.tar_flist[0], 'r') as f:
            self.tar_zdim = f['p'].shape[1] - 6

        self.n_samples = self.nyr * 73 - 1

        self.files = [None for _ in range(self.nyr)]  

    def __len__(self):
        
        return self.n_samples

    def __getitem__(self, global_idx):

        # get year_idx & local_idx
        year_idx = int(global_idx / 73)
        local_idx = int(global_idx % 73)

        # input
        set_x = nc.Dataset(self.inp_flist[year_idx], 'r')['p'][local_idx,:,11:-10]
        set_y = nc.Dataset(self.tar_flist[year_idx], 'r')['p'][local_idx,:self.tar_zdim,11:-10]

        
        # print(global_idx, year_idx, local_idx, set_x.shape, set_y.shape)

        return torch.as_tensor(set_x), torch.as_tensor(set_y)

def get_data_loader(config, inp_path, tar_path, distributed, train):

    dataset = get_data(inp_path, tar_path, train)

    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None

    dataloader = DataLoader(dataset,
                          batch_size=int(config['batch_size']),
                          shuffle=False,
                          pin_memory=torch.cuda.is_available(),
                          num_workers=config['num_workers']
                          )

    if train:
        return dataloader, dataset, sampler
  
    else:
        return dataloader, dataset   

def get_tr_msk(config):

    tmp = nc.Dataset(config['tr_mask'], 'r')['p'][0,:,11:-10] # [15,160,360]

    # expansion for channel domain [15,160,360] -> [62,160,360]
    msk = np.zeros((62,160,360))
    msk[:15] = tmp
    msk[15:30] = tmp
    msk[30:45] = tmp
    msk[45:60] = tmp
    msk[60:] = tmp[0]
    del tmp

    msk = torch.as_tensor(msk).cpu()

    return msk



