import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from data_set_api import IcebergDataset, AdaptedIcebergDataset
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

    
class AngleTrainer:
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
        verbose=1):
        
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
        self.criterion = torch.nn.MSELoss() #BCEWithLogitsLoss()
 
        
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
    
    def get_data_transforms(self, is_train, is_transfer_learning, params={}):
        transform_list = [tv.transforms.ToPILImage()]
        
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
       
    def get_data_loader(self, df, is_train, is_transfer_learning, params={}):
        
        data_transform = self.get_data_transforms(is_train, is_transfer_learning, params)
        
        # initialize our dataset at first
        dataset = AdaptedIcebergDataset(
            df=df,
            transform=data_transform,
            meta_features=['is_iceberg'],
            target='inc_angle'
        )

        # initialize data loader with required number of workers and other params
        data_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=is_train,
                                                   num_workers=0,
                                                   collate_fn=default_collate)
         
        return data_loader
    
    def set_axes_for_loss_and_acc(self):
        plt.ion()
        
        fig = plt.figure(1,figsize=(10,5))
        loss_ax = fig.add_subplot(1,2,1)
        
        loss_ax.set_xlabel('epochs')
        loss_ax.set_ylabel('logloss')
        loss_ax.set_ylim(0,20.)
        
        return fig, loss_ax
        
    def train(self, train_df, valid_df = None, test_df=None, test_ids=None, is_transfer_learning = False, data_augmentation_args={},
        show_cycle=1, do_sa=False, start_epoch=200, t_start=.02, reduce_factor=.9694):
        
        train_data_loader = self.get_data_loader(train_df, True, is_transfer_learning, data_augmentation_args)
        
        if valid_df is not None:
            valid_data_loader = self.get_data_loader(valid_df, False, is_transfer_learning, data_augmentation_args)
        
            
        train_loss_records, valid_loss_records = [], []
        epoch_records = []
        
        fig, loss_ax = self.set_axes_for_loss_and_acc()
        
        best_loss = None
       
        for epoch in range(self.epochs):
            is_show = (epoch+1) % show_cycle == 0
              
            if is_show is True:
                print('epoch=', epoch+1, end=': ')
                
            self.model.train(True)
            train_loss = self.train_epoch(train_data_loader, is_show)
            
            if is_show is True:
                train_loss_records.append(train_loss)
                epoch_records.append(epoch+1)
                
                if valid_df is not None:
                    self.model.train(False)
                        
                    valid_loss = self.valid_epoch(valid_data_loader)
                    valid_loss_records.append(valid_loss)
                    
                    loss_ax.plot(epoch_records, valid_loss_records, 'r*')
                    
                    if best_loss is None or valid_loss < best_loss:
                        best_loss = valid_loss
                        self.save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'best_loss': best_loss,
                            'optimizer' : self.optimizer.state_dict(),
                        })
                        
                        
                    print('Best single model loss: ', best_loss)
                     
                loss_ax.plot(epoch_records, train_loss_records, 'bo')
                
                fig.canvas.draw()
        
        plt.ioff()

    def train_epoch(self, data_loader, is_show=False):
        self.scheduler.step()
        
        train_loss = 0
        counter = 0
        for imgs, other_features, labels in data_loader:
            imgs = Variable(self.to_gpu(imgs))
            other_features = Variable(self.to_gpu(other_features))
            labels = Variable(self.to_gpu(labels))

            outputs = self.model(imgs, other_features)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.data[0]
            counter += 1
        
        train_loss /= counter
        if is_show:
            print('Train set: Average loss: {:.4f}'.format(train_loss)) 
            
            return train_loss
        
        return train_loss
        
    def valid_epoch(self, data_loader):
        # set the model status (train\test)
        
        valid_loss = 0
        counter = 0
        for imgs, other_features, labels in data_loader:
            imgs = Variable(self.to_gpu(imgs), volatile=True)
            other_features = Variable(self.to_gpu(other_features), volatile=True)
            labels = Variable(self.to_gpu(labels), volatile=True)

            outputs = self.model(imgs, other_features)
            loss = self.criterion(outputs, labels)
            
            valid_loss += loss.data[0]
            counter += 1
        
        valid_loss /= counter
        
        print('Valid set: Average loss: {:.4f}'.format(valid_loss)) 
  
        return valid_loss
    
    
    def test(self, test_df, is_transfer_learning = False, is_general_model=False, is_augment=False, data_augmentation_args={}):
        # set the model status (train\test)
        self.model.train(False)
        
        if is_augment is False:
            test_data_transforms = self.get_data_transforms(False, is_transfer_learning, data_augmentation_args)
        else:
            test_data_transforms = self.get_data_transforms(True, is_transfer_learning, data_augmentation_args)
            
        test_dataset = AdaptedIcebergDataset(
            df=test_df,
            transform=test_data_transforms,
            meta_features=['is_iceberg'],
            target='inc_angle'
        )
        
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       collate_fn=default_collate)
                                                   
        # labels are empty in test set
        predictions = []
        for imgs, other_features, _ in test_data_loader:
            imgs = Variable(self.to_gpu(imgs), volatile=True)
            other_features = Variable(self.to_gpu(other_features), volatile=True)
            
            if is_general_model is True:
                outputs = self.model(imgs)
            else:
                outputs = self.model(imgs, other_features)
            
            pred_np = outputs.data.cpu().numpy()
            predictions.append(pred_np)
            #print(pred_np.shape)
            
        predictions = np.concatenate(predictions, axis=0)
                
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