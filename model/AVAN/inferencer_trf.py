import netCDF4 as nc
import numpy as np
from config import config_v01 as config
from AVAN_v01 import AVAN
import torch, os, pathlib, glob

#================================================
# get config
#================================================
config = config()

#================================================
# set lead time
#================================================
lead_time = 40

#================================================
# set input path
#================================================
inp_path = config['test_inp_path']

#================================================
# set output path & name
#================================================
o_path = config['o_path']+config['test_name']
pathlib.Path(o_path).mkdir(parents=True, exist_ok=True)

#================================================
# set checkpoint path
#================================================
if config['tr_period'] == 'pre':
    checkpoint_path = config['o_path']+'PT_checkpoints/best_ckpt.tar'

else:
    checkpoint_path = config['o_path']+'FT_checkpoints/best_ckpt.tar'

#================================================
# set device
#================================================
print(config['local_rank'])
torch.cuda.set_device(config['local_rank'])
torch.backends.cudnn.benchmark = True
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(2) # cpu % limit

#================================================
# get data list
#================================================
inp_flist = glob.glob(inp_path+'*.nc')
inp_flist.sort()
nyr = len(inp_flist)

#================================================
# get stats for reconstruction
#================================================
tmp1 = nc.Dataset(config['test_avg'], 'r')['p'][:,0,0,0]  # [12]
tmp2 = nc.Dataset(config['test_std'], 'r')['p'][:,0,0,0]  # [12]

# expansion for channel domain [10] -> [62]
test_avg, test_std = np.zeros((62)), np.zeros((62))
test_avg[:15]   = tmp1[0] 
test_avg[15:30] = tmp1[1] 
test_avg[30:45] = tmp1[2] 
test_avg[45:60] = tmp1[3] 
test_avg[60:]   = tmp1[4:6] 

test_std[:15]   = tmp2[0] 
test_std[15:30] = tmp2[1] 
test_std[30:45] = tmp2[2] 
test_std[45:60] = tmp2[3] 
test_std[60:]   = tmp2[4:6] 

#================================================
# get land mask
#================================================
tmp = nc.Dataset(config['test_mask'], 'r')['p'][0,:,11:-10,:]  # [lead,160,360]

# expansion for channel domain [lead,160,360] -> [62,160,360]
msk = np.zeros((62,160,360))
msk[:15] = tmp
msk[15:30] = tmp
msk[30:45] = tmp
msk[45:60] = tmp
msk[60:] = tmp[0]
del tmp

msk = torch.as_tensor(msk).to(device, dtype=torch.float)

#================================================
# Inference loop
#================================================
with torch.no_grad():

    count = 0
    for k, inp_file in enumerate(inp_flist, start=0):

        print(count + 1, '/', int(nyr * 73))

        #================================================
        # Load data
        #================================================
        dat = nc.Dataset(inp_file, 'r')['p']
        if k + 1 < len(inp_flist):
            dat2 = nc.Dataset(inp_flist[k+1], 'r')['p']
            
        tdim = dat.shape[0]
        inp_zdim = dat.shape[1]
        tar_zdim = dat.shape[1] - 6
        ydim = dat.shape[2] - 21
        xdim = dat.shape[3]       

        #================================================
        # set params
        #================================================
        config['tdim'] = tdim
        config['inp_zdim'] = inp_zdim
        config['tar_zdim'] = tar_zdim
        config['ydim'] = ydim
        config['xdim'] = xdim
        config['img_size'] = (ydim, xdim)

        if count == 0:

            # load the model
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model = AVAN(config).to(device)
            model.load_state_dict(checkpoint['G_state'])
            model.eval() # set to inference mode

            # expansion for horizontal domain
            test_avg = np.repeat(test_avg.reshape(62,1), ydim*xdim, axis=1).reshape(62,ydim,xdim)
            test_std = np.repeat(test_std.reshape(62,1), ydim*xdim, axis=1).reshape(62,ydim,xdim)

        #================================================
        # integration
        #================================================
        for i in range(tdim):

            # slicing input
            data_slice = torch.as_tensor(dat[i:i+1,:,11:-10]).to(device, dtype=torch.float)

            # set output name
            o_name = 'fcst_'+str(count+1).zfill(4)
        
            fcst = torch.zeros((lead_time,tar_zdim,ydim,xdim)).to(device, dtype=torch.float)
            for j in range(lead_time):

                # inference
                fcst_slice = model(data_slice)

                # land mask out
                fcst_slice = fcst_slice * msk

                fcst[j] = fcst_slice

                # proceed to the next step
                # data_slice = torch.zeros((1,inp_zdim,ydim,xdim)).to(device, dtype=torch.float)
                data_slice = np.zeros((1,inp_zdim,ydim,xdim))
                data_slice[0,:-6] = fcst_slice.cpu().detach().numpy()

                # replacing forcing values as clim
                if i + j < 73:
                    data_slice[0,-6:] = dat[i+j,-6:,11:-10]
                else:
                    data_slice[0,-6:] = dat2[i+j-72,-6:,11:-10]

                # convert to tensor variable
                data_slice = torch.as_tensor(data_slice).to(device, dtype=torch.float)
            
            # reconstruction
            fcst = fcst.cpu().numpy()
            fcst = np.ma.masked_equal(fcst, 0)
            fcst = (fcst * test_std) + test_avg
            fcst = np.array(fcst)

            # save as nc
            fcst.astype('float32').tofile(o_path+o_name+'.gdat')

            ctl = open(o_path+o_name+'.ctl','w')
            ctl.write('dset ^'+o_name+'.gdat\n')
            ctl.write('undef -9.99e+08\n')
            ctl.write('xdef  '+str(xdim)+'  linear   0.   1.\n')
            ctl.write('ydef  '+str(ydim)+'  linear -79.  1.\n')
            ctl.write('zdef  '+str(tar_zdim)+'  linear   1  1\n')
            ctl.write('tdef   '+str(lead_time)+'  linear  jan1979 5dy\n')
            ctl.write('vars   1\n')
            ctl.write('p   '+str(tar_zdim)+'   1  prediction\n')
            ctl.write('ENDVARS\n')
            ctl.close()

            os.system('cdo -s -f nc import_binary '+o_path+o_name+'.ctl '+o_path+o_name+'.nc')
            os.system('rm -f '+o_path+o_name+'.ctl '+o_path+o_name+'.gdat')

            count += 1

