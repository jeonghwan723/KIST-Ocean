import torch.distributed as dist
from AVAN_v01 import AVAN, Disciminator
from config import config_v01 as config
from data_loader import get_data_loader, get_tr_msk
from utils import weighted_l1_loss, lat_weight
import os, pathlib, logging, torch, time, argparse, gc, apex

# get config
config = config()

# set additional config
# for pre-training
if config['tr_period'] == 'pre':
    config['checkpoint_path'] = config['o_path']+'PT_checkpoints/ckpt.tar'
    config['best_checkpoint_path'] = config['o_path']+'PT_checkpoints/best_ckpt.tar'
    config['resuming'] = True if os.path.isfile(config['checkpoint_path']) else False
    pathlib.Path(config['o_path']+'PT_checkpoints').mkdir(parents=True, exist_ok=True)

# for finetuning
else:
    config['pre_checkpoint_path'] = config['o_path']+'PT_checkpoints/best_ckpt.tar'
    config['checkpoint_path'] = config['o_path']+'FT_checkpoints/ckpt.tar'
    config['best_checkpoint_path'] = config['o_path']+'FT_checkpoints/best_ckpt.tar'
    config['resuming'] = True if os.path.isfile(config['pre_checkpoint_path']) else False
    pathlib.Path(config['o_path']+'FT_checkpoints').mkdir(parents=True, exist_ok=True)

# delete cache
gc.collect()
torch.cuda.empty_cache()

# set logging 
logging.basicConfig(
    filename=config['o_path']+'out.log',
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO,
    )

class Trainer():

    def __init__(self, config, global_rank):

        self.config = config
        self.global_rank = global_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        logging.info('rank %d, begin data loader init'%global_rank)

        # set config
        self.tr_inp_path = self.config['tr_inp_path']
        self.tr_tar_path = self.config['tr_tar_path']
        self.val_inp_path = self.config['val_inp_path']
        self.val_tar_path = self.config['val_tar_path']
        self.max_epochs = self.config['max_epochs']

        # Load training & validation dataset & land mask
        self.tr_data_loader, self.tr_dataset, self.tr_sampler = get_data_loader(self.config, self.tr_inp_path, self.tr_tar_path, dist.is_initialized(), train=True)
        self.val_data_loader, self.val_dataset = get_data_loader(self.config, self.val_inp_path, self.val_tar_path, dist.is_initialized(), train=False)
        self.msk = get_tr_msk(config).to(self.device, dtype=torch.float)

        # get dimension inform
        self.config['inp_zdim'] = self.tr_dataset.inp_zdim
        self.config['tar_zdim'] = self.tr_dataset.tar_zdim
        self.config['ydim'] = self.tr_dataset.ydim
        self.config['xdim'] = self.tr_dataset.xdim
        self.config['img_size'] = (self.tr_dataset.ydim, self.tr_dataset.xdim)
        
        logging.info('rank %d, data loader initialized'%global_rank)

        # set model, optimizer, scaler, loss object, set learning rate scheduler
        self.generator = AVAN(self.config).to(self.device)
        self.discriminator = Disciminator(self.config).to(self.device)
        self.optimizer_G = torch.optim.AdamW(self.generator.parameters(), lr=self.config['lr'], weight_decay=1E-1)
        self.optimizer_D = torch.optim.AdamW(self.discriminator.parameters(), lr=self.config['lr'], weight_decay=1E-1)
        self.gscaler_G = torch.cuda.amp.GradScaler()
        self.gscaler_D = torch.cuda.amp.GradScaler()
        self.loss_obj_G = weighted_l1_loss
        self.loss_obj_D = torch.nn.BCELoss()
        self.scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G, factor=0.2, patience=5, mode='min')
        self.scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_D, factor=0.2, patience=5, mode='min')

        # compute latitude weights
        self.latw = lat_weight(self.config['ydim'], self.config['lat1'], self.config['lat2']).to(self.device, dtype=torch.float)

        if torch.distributed.is_initialized():
            self.generator = torch.nn.parallel.DistributedDataParallel(
                self.generator,
                device_ids=[self.config.local_rank],
                output_device=[self.config.local_rank],
                find_unused_parameters=True,
                )
            
            self.discriminator = torch.nn.parallel.DistributedDataParallel(
                self.discriminator,
                device_ids=[self.config.local_rank],
                output_device=[self.config.local_rank],
                find_unused_parameters=True,
                )

        self.iters = 0
        self.startEpoch = 0

        # If a checkpoint file exists, restore the parameters and continue learning.
        if config['tr_period'] == 'pre': 
            if os.path.isfile(config['checkpoint_path']) == True:
                self.restore_checkpoint(config['checkpoint_path'])
        
        else:
            self.restore_checkpoint(config['pre_checkpoint_path'])
            self.iters = 0
            self.startEpoch = 0            

        self.epoch = self.startEpoch

    def train(self):

        print("Starting Training Loop...")
        
        best_valid_loss = 1.e6
        for epoch in range(self.startEpoch, self.max_epochs):

            # print('Epoch: ', epoch)

            if torch.distributed.is_initialized():                
                self.train_sampler.set_epoch(epoch)

            start = time.time()

            train_logs = self.tr_proc()
            valid_logs = self.val_proc()

            self.scheduler_G.step(valid_logs['valid_loss'])
            self.scheduler_D.step(valid_logs['valid_loss'])

            self.save_checkpoint(self.config['checkpoint_path'])

            if self.global_rank == 0:
                self.save_checkpoint(self.config['checkpoint_path'])

                if valid_logs['valid_loss'] <= best_valid_loss:
                    self.save_checkpoint(self.config['best_checkpoint_path'])
                    best_valid_loss = valid_logs['valid_loss']
            
            logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            logging.info('Train loss: {}. Valid loss: {}'.format(train_logs['G loss'], valid_logs['valid_loss']))
            print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            print('Train loss: {}. Valid loss: {}'.format(train_logs['G loss'], valid_logs['valid_loss']))

    # set training process
    def tr_proc(self):

        self.epoch += 1
        self.generator.train()
        self.discriminator.train()

        # iteration (mini batch)
        for i, data in enumerate(self.tr_data_loader, 0):

            self.iters += 1
            inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
            tdim = inp.shape[0]

            # make discriminator label
            label_real = torch.ones(tdim, self.config['tar_zdim'], 10, 22).to(self.device)
            label_fake = torch.zeros(tdim, self.config['tar_zdim'], 10, 22).to(self.device)

            ### Generator
            with torch.cuda.amp.autocast(self.config['enable_amp']):

                # inference
                yhat = self.generator(inp).to(self.device, dtype = torch.float)

                # land maskout
                yhat *= self.msk
                
                # judgement of discriminator
                out_fake = self.discriminator(yhat)

                # loss
                loss_G_recon = self.loss_obj_G(yhat, tar, self.latw)
                # loss_G_adv = self.loss_obj_D(out_fake, label_real)
                loss_G_adv = torch.nn.functional.binary_cross_entropy_with_logits(out_fake, label_real)
                # loss_G = (loss_G_recon * 0.8) + (loss_G_adv * 0.2)
                loss_G = (loss_G_recon * 0.6) + (loss_G_adv * 0.4)
                # loss_G = (loss_G_recon * 0.4) + (loss_G_adv * 0.6)

            # parameter update (back propagation)
            self.gscaler_G.scale(loss_G).backward()
            self.gscaler_G.step(self.optimizer_G)
            self.gscaler_G.update()


            ### Discriminator
            with torch.cuda.amp.autocast(self.config['enable_amp']):

                out_fake = self.discriminator(yhat.detach())
                out_real = self.discriminator(tar)

                # loss
                # loss_D_adv1 = self.loss_obj_D(out_fake, label_fake)
                # loss_D_adv2 = self.loss_obj_D(out_real, label_real)
                loss_D_adv1 = torch.nn.functional.binary_cross_entropy_with_logits(out_fake, label_fake)
                loss_D_adv2 = torch.nn.functional.binary_cross_entropy_with_logits(out_real, label_real)
                loss_D = (loss_D_adv1 + loss_D_adv2)

            # parameter update (back propagation)
            self.gscaler_D.scale(loss_D).backward()
            self.gscaler_D.step(self.optimizer_D)
            self.gscaler_D.update()

            logs = {'G loss': loss_G, 'D loss': loss_D}

            if i % 50 == 0:
                print(f"    G Loss: {loss_G.item():.4f} D Loss: {loss_D.item():.4f}")

            if torch.distributed.is_initialized():
                for key in sorted(logs.keys()):
                    torch.distributed.all_reduce(logs[key].detach())
                    logs[key] = float(logs[key]/torch.distributed.get_world_size())

        return logs

    # set validation process
    def val_proc(self):

        self.generator.eval()

        valid_buff = torch.zeros((3), dtype=torch.float32, device=self.device)
        valid_loss = valid_buff[0].view(-1)
        valid_l1 = valid_buff[1].view(-1)
        valid_steps = valid_buff[2].view(-1)        

        with torch.no_grad():
            for i, data in enumerate(self.val_data_loader, 0):

                # get data
                inp, tar  = map(lambda x: x.to(self.device, dtype = torch.float), data)

                # inference
                yhat = self.generator(inp).to(self.device, dtype = torch.float)

                # land mask out
                yhat *= self.msk

                valid_loss += self.loss_obj_G(yhat, tar, self.latw) 
                valid_l1 += torch.nn.functional.l1_loss(yhat, tar)
                valid_steps += 1.

        if dist.is_initialized():
            dist.all_reduce(valid_buff)
        
        # divide by number of steps
        valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]

        valid_buff_log = valid_buff.detach().cpu().numpy()

        logs = {'valid_l1': valid_buff_log[1], 
                'valid_loss': valid_buff_log[0]}
        
        return logs

    # save model
    def save_checkpoint(self, checkpoint_path, generator=None, discriminator=None,):

        if not generator:
            generator = self.generator

        if not discriminator:
            discriminator = self.discriminator

        torch.save({'iters': self.iters, 
                    'epoch': self.epoch, 
                    'G_state': generator.state_dict(),
                    'D_state': discriminator.state_dict(),
                    'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                    'optimizer_D_state_dict': self.optimizer_D.state_dict()}, 
                    checkpoint_path)


    def restore_checkpoint(self, checkpoint_path):

        checkpoint = torch.load(checkpoint_path, 
                                map_location='cuda:{}'.format(self.config['local_rank']))
        
        self.generator.load_state_dict(checkpoint['G_state'])
        self.discriminator.load_state_dict(checkpoint['D_state'])
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

if __name__ == '__main__':

    # # set ArgmentParser
    # parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument('--global_rank', type=int, default=0)
    # parser.add_argument('--num_workers', type=int, default=24)
    # parser.add_argument("--local_rank", type=int,
    #                 help="Local rank. Necessary for using the torch.distributed.launch utility.")
    # parser.add_argument('--world_size', type=int, default=0)

    # get global_rank, local_rank, world_size
    # global_rank = int(os.environ['RANK'])
    local_rank = config['local_rank']

    world_size = 1
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])

    if world_size > 1:
        dist.init_process_group(backend='nccl')

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    global_rank = 0

    config['world_size'] = 1
    if 'WORLD_SIZE' in os.environ:
        config['world_size'] = int(os.environ['WORLD_SIZE'])

    trainer = Trainer(config, global_rank)
    trainer.train()
    logging.info('DONE ---- rank %d'%global_rank)



