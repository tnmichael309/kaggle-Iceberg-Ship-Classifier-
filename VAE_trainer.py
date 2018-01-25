import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from data_set_api import IcebergDataset, AdaptedIcebergDataset, VAEIcebergDataset
from torch.utils.data.dataloader import default_collate
import torchvision as tv
import pandas as pd
import numpy as np
from random_affine_api import RandomAffineTransform
import math
import matplotlib
import matplotlib.pyplot as plt
import time
from torchsample.transforms import TypeCast, RangeNormalize, RandomAffine
import torch.nn as nn
import copy
from sklearn.metrics import log_loss
from scipy.stats import gmean


def loss_function(recon_x, x, mu, logvar, batch_size):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 5625))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size * 5625

    return BCE, KLD
    
class VAETrainer:
    def __init__(self, 
        model,
        optimizer = None,
        epochs=300,
        milestones=[60, 160, 260],
        gamma=0.1,
        batch_size=256, 
        use_cuda=False, 
        gpu_idx=0,
        seed = 0,
        best_model_name='model_best.pth.tar',
        verbose=1,
        crop_size=75,
        start_valid_ensemble_epoch=50):
        
        self.use_cuda = use_cuda
        self.gpu_idx = gpu_idx
        self.best_model_name = best_model_name
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.use_cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.set_device(gpu_idx)
            print("gpu:",gpu_idx," available:", torch.cuda.is_available())
            
            with torch.cuda.device(gpu_idx):
                self.model = self.to_gpu(model)
        else:
            self.model = self.from_gpu(model)
        
        if self.model is None:
            raise ValueError('Cannot initialize the model')
         
        if verbose > 0:
            print("Model in use:")
            print(model)
        
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.set_optimizer(optimizer)
        self.milestones = milestones
        self.set_scheduler(milestones, gamma)
        self.crop_size = crop_size
        
    # I add this for fine_tune capabilities, where we can provided a new optimizer on only 
    # specific parameters to update
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_scheduler(self, milestones, gamma):
        if self.optimizer is None:
            print("No optimizer provided")
        else:
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones, gamma=gamma)
        
    def to_gpu(self, tensor):
        if self.use_cuda:
            return tensor.cuda(self.gpu_idx)
        else:
            return tensor

    def from_gpu(self, tensor):
        if self.use_cuda:
            return tensor.cpu()
        else:
            return tensor

    
    '''
    def get_data_transforms(self, is_train, is_transfer_learning, params={}):
        transform_list = [tv.transforms.ToPILImage(mode='L'), tv.transforms.CenterCrop(self.crop_size)]
        
        if is_transfer_learning is True:
            transform_list.append(tv.transforms.Resize((197,197)))
        
        if 'crop' in params and params['crop'] is True:
            if is_train is True:
                transform_list.append(tv.transforms.RandomCrop(64))
            else:
                transform_list.append(tv.transforms.CenterCrop(64))
                
        if is_train is True:
            if 'mirror' in params and params['mirror'] is True:
                transform_list.append(tv.transforms.RandomVerticalFlip())
                transform_list.append(tv.transforms.RandomHorizontalFlip())
             
            rotation_range = None
            scale_range = None
            translation_range = None
            
            if 'rotate' in params and params['rotate'] is True:
                rotation_range = (-math.pi*15./180., math.pi*15./180.)

            if 'scale' in params and params['scale'] is True:
                scale_range = (0.9, 1.1)
                
            if 'translation' in params and params['translation'] is True:
                translation_range = (-0.1, 0.1)    
             
            to_do_affine = (rotation_range is not None) or (scale_range is not None) or (translation_range is not None)
            if to_do_affine is True:
                affine_transform = RandomAffineTransform(rotation_range=rotation_range, scale_range=scale_range, translation_range=translation_range)
                transform_list.append(tv.transforms.Lambda(lambda img: affine_transform(img))) 
            
            if 'color jitter' in params and params['color jitter'] is True:
                transform_list.append(tv.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
        
        
                
        transform_list.append(tv.transforms.ToTensor())
        #print("Using:\n", transform_list)
        
        return tv.transforms.Compose(transform_list)
    '''
    
    def get_data_transforms(self, is_train, is_transfer_learning, params={}):
        transform_list = [tv.transforms.ToPILImage(), tv.transforms.CenterCrop(self.crop_size), tv.transforms.ToTensor()]
        
        if is_transfer_learning is True:
            transform_list.append(tv.transforms.Resize((197,197)))
        
        transform_list.append(tv.transforms.Lambda(lambda img: RangeNormalize(0, 1)(TypeCast('float')(img))))
        
        if is_train is True:
            if 'mirror' in params and params['mirror'] is True:
                transform_list.append(tv.transforms.RandomVerticalFlip())
                transform_list.append(tv.transforms.RandomHorizontalFlip())
             
            rotation_range = None
            scale_range = None
            translation_range = None
            
            if 'rotate' in params and params['rotate'] is True:
                rotation_range = 15

            if 'scale' in params and params['scale'] is True:
                scale_range = (0.9, 1.1)
                
            if 'translation' in params and params['translation'] is True:
                translation_range = (-0.1, 0.1)    
             
            to_do_affine = (rotation_range is not None) or (scale_range is not None) or (translation_range is not None)
            if to_do_affine is True:
                affine_transform = RandomAffine(rotation_range=rotation_range, zoom_range=scale_range, translation_range=translation_range)
                transform_list.append(tv.transforms.Lambda(lambda img: affine_transform(img))) 
            
            if 'color jitter' in params and params['color jitter'] is True:
                transform_list.append(tv.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
        
        #print("Using:\n", transform_list)
        
        return tv.transforms.Compose(transform_list)
        
    def get_data_loader(self, df, is_train, is_transfer_learning, params={}):
        
        data_transform = self.get_data_transforms(is_train, is_transfer_learning, params)
        
        # initialize our dataset at first
        dataset = VAEIcebergDataset(
            df=df,
            transform=data_transform
        )

        # initialize data loader with required number of workers and other params
        data_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=is_train,
                                                   num_workers=0,
                                                   collate_fn=default_collate)
         
        return data_loader
    
    def set_axes_for_loss_and_acc(self):
        #plt.ion()
        return None, None, None
        
        fig = plt.figure(1,figsize=(10,5))
        loss_ax = fig.add_subplot(1,2,1)
        acc_ax = fig.add_subplot(1,2,2)
        
        loss_ax.set_xlabel('epochs')
        loss_ax.set_ylabel('logloss')
        acc_ax.set_xlabel('epochs')
        acc_ax.set_ylabel('accuracy')
        loss_ax.set_ylim(0,1.)
        acc_ax.set_ylim(0.4,1.)
        
        return fig, loss_ax, acc_ax
        
    def train(self, train_df, valid_df = None, test_df=None, test_ids=None, is_transfer_learning = False, data_augmentation_args={},
        show_cycle=1, do_sa=False, start_epoch=200, t_start=.02, reduce_factor=.9694):
        
        train_data_loader = self.get_data_loader(train_df, True, is_transfer_learning, data_augmentation_args)
        
        if valid_df is not None:
            valid_data_loader = self.get_data_loader(valid_df, False, is_transfer_learning, data_augmentation_args)
            self.valid_ensemble_df = pd.DataFrame(data=np.zeros((valid_df.shape[0],1)), columns=['f_0'])
            print(self.valid_ensemble_df.shape)
         
        if test_df is not None:
            self.test_ensemble_df = pd.DataFrame(data=np.zeros((test_df.shape[0],1)), columns=['f_0'])
            print(self.test_ensemble_df.shape)
            
        train_loss_records, valid_loss_records = [], []
        train_kl_div_records, valid_kl_div_records = [], []
        epoch_records = []
        
        fig, loss_ax, acc_ax = self.set_axes_for_loss_and_acc()
        
        best_loss = None
       
        for epoch in range(self.epochs):
            is_show = (epoch+1) % show_cycle == 0
              
            if is_show is True:
                print('epoch=', epoch+1, end=': ')
                
            self.model.train(True)
            train_loss, train_kl_div = self.train_epoch(train_data_loader, is_show)
            
            if is_show is True:
                train_loss_records.append(train_loss)
                train_kl_div_records.append(train_kl_div)
                epoch_records.append(epoch+1)
                
                if valid_df is not None:
                    self.model.train(False)
                       
                    valid_loss, valid_kl_div = self.valid_epoch(valid_data_loader)
                    valid_loss_records.append(valid_loss)
                    valid_kl_div_records.append(valid_kl_div)
                    
                    #loss_ax.plot(epoch_records, valid_loss_records, 'r*')
                    #acc_ax.plot(epoch_records, valid_kl_div_records, 'r*')
                
                    if best_loss is None or valid_loss < best_loss:
                        best_loss = valid_loss
                        self.save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'best_loss': best_loss,
                            'optimizer' : self.optimizer.state_dict(),
                        })
                        
                        
                    print('Best single model loss: ', best_loss)
                    
                    if (epoch+1) in self.milestones:
                        pass #self.load_checkpoint()

                    if do_sa is True:
                        if epoch > start_epoch:
                            self.perform_simulated_annealing(valid_df, valid_loss, data_augmentation_args, t_start)
                        
                        t_start = t_start * reduce_factor
                        
                #loss_ax.plot(epoch_records, train_loss_records, 'bo')
                #acc_ax.plot(epoch_records, train_kl_div_records, 'bo')
                
                #fig.canvas.draw()
        
        #plt.ioff()
        
        
    def train_epoch(self, data_loader, is_show=False):
        self.scheduler.step()
        
        recnt_loss = 0
        kl_loss = 0
        counter = 0
        for imgs in data_loader:
            
            imgs = Variable(self.to_gpu(imgs.view(imgs.size(0), -1))) 
            out, mu, log_var = self.model(imgs)
            
            reconst_loss, kl_divergence = loss_function(out, imgs, mu, log_var, len(imgs))
            #reconst_loss = F.binary_cross_entropy(out, imgs, size_average=False)
            #kl_divergence = torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var -1))
        
            # Backprop + Optimize
            total_loss = reconst_loss + kl_divergence
        
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            recnt_loss += reconst_loss.data[0]
            kl_loss += kl_divergence.data[0]
            counter += 1
            
        recnt_loss /= counter
        kl_loss /= counter
        
        if is_show:
            print('Train set: Reconstruct loss: {:.6f}, KL Div loss: {:.6f}'.format(recnt_loss, kl_loss))
        
        return recnt_loss+kl_loss, recnt_loss
        
        
    def valid_epoch(self, data_loader):
        recnt_loss = 0
        kl_loss = 0
        counter = 0
        
        for imgs in data_loader:
            
            imgs = Variable(self.to_gpu(imgs.view(imgs.size(0), -1)), volatile=True) 
            out, mu, log_var = self.model(imgs)
            
            reconst_loss, kl_divergence = loss_function(out, imgs, mu, log_var, len(imgs))
            #reconst_loss = F.binary_cross_entropy(out, imgs, size_average=False)
            #kl_divergence = torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var -1))
            
            recnt_loss += reconst_loss.data[0]
            kl_loss += kl_divergence.data[0]
            counter += 1
            
        recnt_loss /= counter
        kl_loss /= counter
        
        print('Valid set: Reconstruct loss: {:.6f}, KL Div loss: {:.6f}'.format(recnt_loss, kl_loss))
        
        return recnt_loss+kl_loss, recnt_loss
    
    def test(self, test_df, is_transfer_learning = False, is_general_model=False, is_augment=False, data_augmentation_args={}):
        # set the model status (train\test)
        self.model.train(False)
        
        if is_augment is False:
            test_data_transforms = self.get_data_transforms(False, is_transfer_learning, data_augmentation_args)
        else:
            test_data_transforms = self.get_data_transforms(True, is_transfer_learning, data_augmentation_args)
            
        test_dataset = VAEIcebergDataset(
            df=test_df,
            transform=test_data_transforms
        )
        
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       collate_fn=default_collate)
                                                   
        # labels are empty in test set
        predictions = []
        for imgs in test_data_loader:
        
            imgs = Variable(self.to_gpu(imgs.view(imgs.size(0), -1)), volatile=True) 
            out, _, _ = self.model(imgs)
            
            out = out.view(out.size(0), 1, self.crop_size, self.crop_size)
            pred_np = out.data.cpu().numpy()
            
            for pic in pred_np:
                pic = pic.reshape((self.crop_size,self.crop_size))
                predictions.append(pic)
            #print(pred_np.shape)
                   
        return predictions
    
    def save_checkpoint(self, state):
        torch.save(state, self.best_model_name)

    def load_checkpoint(self, filename=None):
        if filename is None:
            filename = self.best_model_name
            
        state = torch.load(filename)
        print('epoch=', state['epoch'], 'best_loss=', state['best_loss'])
        self.model.load_state_dict(state['state_dict'])
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))