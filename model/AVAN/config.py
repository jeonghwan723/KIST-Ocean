
#======================================================
# Adversarial VAN version 1
#======================================================
def config_v01():

    config = {}

    # machine settings
    config['local_rank'] = 3  # GPU number
    config['num_workers'] = 4  # number of CPU cores

    # training settings
    config['tr_period'] = 'pre'   # 'pre' for pre-training or 'fine' for finetuning
    config['batch_size'] = 20
    config['max_epochs'] = 50
    
    # set architecture
    config['lr'] = 1E-4
    config['num_stages'] = 3               # number of stages
    config['depths'] = [2, 2, 2]           # depth for each stage
    config['embed_dims'] = [128, 256, 256] # embeding dimensions for each stage
    config['drop_path_rate'] = 0.0         # drop path rate
    config['mlp_ratios'] = [1, 1, 1]       # the number of neurons at MLP layer
    config['drop_rate'] = 0.0              # dropout rate
    
    #======================================================
    # input & target files
    #======================================================
    # config['tr_inp_path'] = '/home/jhkim/task/ocean2/model/dataset/CESM2_tr_inp/'
    # config['tr_tar_path'] = '/home/jhkim/task/ocean2/model/dataset/CESM2_tr_tar/'
    # config['val_inp_path'] = '/home/jhkim/task/ocean2/model/dataset/CESM2_val_inp/'
    # config['val_tar_path'] = '/home/jhkim/task/ocean2/model/dataset/CESM2_val_tar/'
    # config['test_inp_path'] = '/home/jhkim/task/ocean2/model/dataset/CESM2_val_inp/'

    config['tr_inp_path'] = '/home/jhkim/task/ocean2/model/dataset/GODAS_tr_inp/'
    config['tr_tar_path'] = '/home/jhkim/task/ocean2/model/dataset/GODAS_tr_tar/'
    config['val_inp_path'] = '/home/jhkim/task/ocean2/model/dataset/GODAS_val_inp/'
    config['val_tar_path'] = '/home/jhkim/task/ocean2/model/dataset/GODAS_val_tar/'
    config['test_inp_path'] = '/home/jhkim/task/ocean2/model/dataset/GODAS_val_inp/'

    config['lat1'] = -79    # southern bound of latitude
    config['lat2'] = 80     # northern bound of latitude

    #======================================================
    # Stats for reconstruction in inference (i.e., standardized -> original)
    #======================================================
    # config['test_avg'] = '/home/jhkim/task/ocean2/data/CESM2_AVG_013.nc'
    # config['test_std'] = '/home/jhkim/task/ocean2/data/CESM2_STD_013.nc'
    # config['test_clim'] = '/home/jhkim/task/ocean2/data/CESM2_CLIM_FRC.nc'

    config['test_avg'] = '/home/jhkim/task/ocean2/data/OBS_AVG.nc'
    config['test_std'] = '/home/jhkim/task/ocean2/data/OBS_STD.nc'
    config['test_clim'] = '/home/jhkim/task/ocean2/data/OBS_CLIM_FRC.nc'

    #======================================================
    # Land mask files (0: land, 1: ocean)
    #======================================================
    # config['tr_mask'] = '/home/jhkim/task/ocean2/data/Landmask_CESM2.nc'
    # config['val_mask'] = '/home/jhkim/task/ocean2/data/Landmask_CESM2.nc'
    # config['test_mask'] = '/home/jhkim/task/ocean2/data/Landmask_CESM2.nc'

    config['tr_mask'] = '/home/jhkim/task/ocean2/data/Landmask_GODAS.nc'
    config['val_mask'] = '/home/jhkim/task/ocean2/data/Landmask_GODAS.nc'
    config['test_mask'] = '/home/jhkim/task/ocean2/data/Landmask_GODAS.nc'

    #======================================================
    # output path
    #======================================================
    config['o_path'] = '/home/jhkim/task/ocean2/model/output/AVAN_v01_0.6/'
    config['test_name'] = 'FCST_GODAS_trf/'

    #======================================================
    # enable or disable AMP (i.e., mixed precision learning)
    #======================================================
    config['enable_amp'] = True

    return config

