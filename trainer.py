import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from data_set_api import IcebergDataset, PLIcebergDataset
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

    
class Trainer:
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
        
        self.criterion = torch.nn.BCELoss() #BCEWithLogitsLoss()
           
        self.valid_ensemble_df = pd.DataFrame()
        self.valid_ensemble_info = {
            'min_loss': None,
            'best_column': None,
            'col_count': 0
        }
        self.start_valid_ensemble_epoch = start_valid_ensemble_epoch
        
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
            transform_list.append(tv.transforms.Resize((224,224)))
        
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
        transform_list = [tv.transforms.ToTensor()]
        
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
    '''    
    def get_data_loader(self, df, is_train, is_transfer_learning, params={}):
        
        data_transform = self.get_data_transforms(is_train, is_transfer_learning, params)
        
        # initialize our dataset at first
        dataset = IcebergDataset(
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
    
    def get_pl_data_loader(self, df, is_train, is_transfer_learning, params={}):
        
        data_transform = self.get_data_transforms(is_train, is_transfer_learning, params)
        
        # initialize our dataset at first
        dataset = PLIcebergDataset(
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
        plt.ion()
        
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
        show_cycle=1, do_sa=False, start_epoch=200, t_start=.02, reduce_factor=.9694,
        pl_enabled=False, pl_beta=.8, pl_beta_end=.2, pl_dec_start=160, pl_dec_end=300):
        
        if pl_enabled is False:
            train_data_loader = self.get_data_loader(train_df, True, is_transfer_learning, data_augmentation_args)
        else:
            train_data_loader = self.get_pl_data_loader(train_df, True, is_transfer_learning, data_augmentation_args)
        
        
        if valid_df is not None:
            valid_data_loader = self.get_data_loader(valid_df, False, is_transfer_learning, data_augmentation_args)
            self.valid_ensemble_df = pd.DataFrame(data=np.zeros((valid_df.shape[0],1)), columns=['f_0'])
            print(self.valid_ensemble_df.shape)
         
        if test_df is not None:
            self.test_ensemble_df = pd.DataFrame(data=np.zeros((test_df.shape[0],1)), columns=['f_0'])
            print(self.test_ensemble_df.shape)
            
        train_loss_records, valid_loss_records = [], []
        train_acc_records, valid_acc_records = [], []
        epoch_records = []
        
        fig, loss_ax, acc_ax = self.set_axes_for_loss_and_acc()
        
        best_loss = None
       
        for epoch in range(self.epochs):
            is_show = (epoch+1) % show_cycle == 0
              
            if is_show is True:
                print('epoch=', epoch+1, end=': ')
                
            self.model.train(True)
            
            if pl_enabled is False:
                train_loss, train_acc = self.train_epoch(train_data_loader, is_show)
            else:
                pl_beta_used = None
                if epoch <= pl_dec_start:
                    pl_beta_used = pl_beta
                elif epoch > pl_dec_start and epoch <= pl_dec_end:
                    pl_beta_used = pl_beta + (pl_beta_end-pl_beta)*(epoch-pl_dec_start) / (1. * (pl_dec_end - pl_dec_start))
                else:
                    pl_beta_used = pl_beta_end
                 
                print('pl beta=', pl_beta_used)
                train_loss, train_acc = self.pl_train_epoch(train_data_loader, is_show, pl_beta_used)
                
            if is_show is True:
                train_loss_records.append(train_loss)
                train_acc_records.append(train_acc)
                epoch_records.append(epoch+1)
                
                if valid_df is not None:
                    self.model.train(False)
                    
                    if (epoch+1 >= self.start_valid_ensemble_epoch):
                        do_valid_ensemble = False
                    else:
                        do_valid_ensemble = False
                        
                    valid_loss, valid_acc = self.valid_epoch(valid_data_loader, do_valid_ensemble, test_df)
                    valid_loss_records.append(valid_loss)
                    valid_acc_records.append(valid_acc)
                    
                    loss_ax.plot(epoch_records, valid_loss_records, 'r*')
                    acc_ax.plot(epoch_records, valid_acc_records, 'r*')
                
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
                        
                loss_ax.plot(epoch_records, train_loss_records, 'bo')
                acc_ax.plot(epoch_records, train_acc_records, 'bo')
                
                fig.canvas.draw()
        
        plt.ioff()
        
        if test_df is not None and self.valid_ensemble_info['col_count'] > 0:
            print("Submission generated, ensemble {} models in total".format(self.valid_ensemble_info['col_count']))
            submission = pd.DataFrame()
            submission['id'] = test_ids
            submission['is_iceberg'] = self.test_ensemble_df['is_iceberg']
            submission_name = self.best_model_name.replace('Trained_model/', 'Submissions/').replace('.db', '_submission.csv')
            submission.to_csv(submission_name, float_format='%.15f', index=False)
            
        
    def pl_train_epoch(self, data_loader, is_show=False, pl_beta=.8):
        self.scheduler.step()
        
        train_loss = 0
        correct = 0
        counter = 0
        for imgs, other_features, labels, is_pseudo in data_loader:
            imgs = Variable(self.to_gpu(imgs))
            other_features = Variable(self.to_gpu(other_features))
            labels = Variable(self.to_gpu(labels))
            is_pseudo = Variable(self.to_gpu(is_pseudo))

            
            outputs = self.model(imgs, other_features)
            
            # paper pl loss (hard loss)
            predictions = torch.ge(outputs, .5).type(torch.cuda.FloatTensor)
            
            is_pseudo_mask = torch.eq(is_pseudo, 1.0)
            pl_outputs = torch.masked_select(outputs, is_pseudo_mask)
            pl_labels = torch.masked_select(labels, is_pseudo_mask)
            pl_predictions = torch.masked_select(predictions, is_pseudo_mask)
            loss1 = nn.BCELoss(size_average=False)(pl_outputs, pl_labels)
            loss2 = nn.BCELoss(size_average=False)(pl_outputs, pl_predictions)
            pl_loss = torch.add(torch.mul(loss1, pl_beta), torch.mul(loss2, 1.-pl_beta))
            
            non_pseudo_mask = torch.eq(is_pseudo, 0.0)
            npl_outputs = torch.masked_select(outputs, non_pseudo_mask)
            npl_labels = torch.masked_select(labels, non_pseudo_mask)
            npl_loss = nn.BCELoss(size_average=False)(npl_outputs, npl_labels)
            
            loss = torch.div(torch.add(pl_loss, npl_loss), outputs.nelement())
            
            self.optimizer.zero_grad()
            
            loss.backward()
            nn.utils.clip_grad_norm(self.model.parameters(), 2.)
            self.optimizer.step()
            
            train_loss += loss.data[0]
            preds_np = outputs.data.cpu().numpy()
            labels_np = labels.data.cpu().numpy()
            preds_np = np.array([1.0 if p >= 0.5 else 0.0 for p in preds_np])
            labels_np = labels_np.reshape((labels_np.shape[0],))
            labels_np = np.array([1.0 if p >= 0.5 else 0.0 for p in labels_np])
            correct += (preds_np == labels_np).sum()
            counter += 1
        
        train_loss /= counter
        accuracy = correct / len(data_loader.dataset)
        if is_show:
            print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                train_loss, correct, len(data_loader.dataset),
                100. * accuracy)) 
            
            return train_loss, accuracy
        
        return train_loss, accuracy
    
    def train_epoch(self, data_loader, is_show=False):
        self.scheduler.step()
        
        train_loss = 0
        correct = 0
        counter = 0
        for imgs, other_features, labels in data_loader:
            
            imgs = Variable(self.to_gpu(imgs))
            other_features = Variable(self.to_gpu(other_features))
            labels = Variable(self.to_gpu(labels))
            
            outputs = self.model(imgs, other_features)
            
            # to be modified
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            
            loss.backward()
            nn.utils.clip_grad_norm(self.model.parameters(), 2.)
            
            self.optimizer.step()
            
            train_loss += loss.data[0]
            preds_np = outputs.data.cpu().numpy()
            labels_np = labels.data.cpu().numpy()
            preds_np = np.array([1.0 if p >= 0.5 else 0.0 for p in preds_np])
            labels_np = labels_np.reshape((labels_np.shape[0],))
            correct += (preds_np == labels_np).sum()
            counter += 1
        
        train_loss /= counter
        accuracy = correct / len(data_loader.dataset)
        if is_show:
            print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                train_loss, correct, len(data_loader.dataset),
                100. * accuracy)) 
            
            return train_loss, accuracy
        
        return train_loss, accuracy
        
    def valid_epoch(self, data_loader, is_ensemble=False, test_df=None):
        # set the model status (train\test)
        
        valid_loss = 0
        correct = 0
        counter = 0
        predictions = []
        answers = []
        
        for imgs, other_features, labels in data_loader:
            imgs = Variable(self.to_gpu(imgs), volatile=True)
            other_features = Variable(self.to_gpu(other_features), volatile=True)
            labels = Variable(self.to_gpu(labels), volatile=True)

            outputs = self.model(imgs, other_features)
            loss = self.criterion(outputs, labels)
            
            valid_loss += loss.data[0]
            preds_np = outputs.data.cpu().numpy()
            labels_np = labels.data.cpu().numpy()
            preds_np = preds_np.reshape((preds_np.shape[0],))
            labels_np = labels_np.reshape((labels_np.shape[0],))
            
            predictions.append(preds_np)
            answers.append(labels_np)
            
            preds_np = np.array([1.0 if p >= 0.5 else 0.0 for p in preds_np])
            correct += (preds_np == labels_np).sum()
            counter += 1
        
        valid_loss /= counter
        accuracy = correct / len(data_loader.dataset)
        
        if is_ensemble is True and valid_loss <= .23:
            predictions = np.concatenate(predictions, axis=0)
            answers = np.concatenate(answers, axis=0)
            self.ensemble_valid(predictions, answers, test_df=test_df)
            
        print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            valid_loss, correct, len(data_loader.dataset),
            100. * accuracy)) 
  
        return valid_loss, accuracy
    
    def ensemble_valid(self, new_pred, answer, cutoff_lo=.8, cutoff_hi=.2, test_df=None):
        '''
        self.valid_ensemble_df = pd.DataFrame()
        self.valid_ensemble_info = {
            'min_loss': 1.0,
            'best_column': None
        }
        '''
        col_name = 'f_' + str(self.valid_ensemble_info['col_count'])
        
        new_pred_loss = log_loss(answer, new_pred)
        self.valid_ensemble_info['col_count'] += 1
        
        if self.valid_ensemble_info['min_loss'] is None or new_pred_loss < self.valid_ensemble_info['min_loss']:
            self.valid_ensemble_info['min_loss'] = new_pred_loss
            self.valid_ensemble_info['best_column'] = col_name
            
        self.valid_ensemble_df = self.ensemble_df(self.valid_ensemble_df, col_name, new_pred, cutoff_lo=cutoff_lo, cutoff_hi=cutoff_hi)
        loss = log_loss(answer, self.valid_ensemble_df['is_iceberg'])
        print('Valid set ensemble loss: {:.4f}'.format(loss))
        
        if test_df is not None:
            self.test_ensemble_df = self.ensemble_df(self.test_ensemble_df, col_name, self.test(test_df), cutoff_lo=cutoff_lo, cutoff_hi=cutoff_hi)
        

    def ensemble_df(self, df, col_name, pred, cutoff_lo=.8, cutoff_hi=.2):
        df[col_name] = pred
        
        col_num = self.valid_ensemble_info['col_count']
        cols = ['f_'+str(i) for i in range(col_num)]
        
        best_col = self.valid_ensemble_info['best_column']
        
        
        df['is_iceberg'] = df.loc[:, cols].mean(axis=1)
        
        '''
        df['mean'] = df.loc[:, cols].mean(axis=1)
        df['gmean'] = gmean(df.loc[:, cols], axis=1)
        df['max'] = df.loc[:, cols].max(axis=1)
        df['min'] = df.loc[:, cols].min(axis=1)
        
        df['is_iceberg'] = np.where(np.all(df.loc[:, cols] > cutoff_lo, axis=1),
                                            df['max'],
                                            
                                            np.where(np.all(df.loc[:, cols] < cutoff_hi, axis=1),
                                                df['min'],
                                                
                                                np.where(df[best_col] > .5,
                                                    df['mean'],
                                                    df['gmean'],    
                                                )
                                            )
                                    )
        '''
        return df
        
    def perform_simulated_annealing(self, valid_df, cur_loss, data_augmentation_args, T):
        y = valid_df['is_iceberg']
        old_model = copy.deepcopy(self.model)
        
        for _ in range(20):
            del self.model
            self.model = self._get_nn_neighbor(copy.deepcopy(old_model))
            pred_y = self.test(valid_df, False, False, False, data_augmentation_args)
            
            avg_loss = log_loss(y, pred_y)
            
            if avg_loss <= cur_loss:
                #print(old_model.state_dict())
                del old_model
                old_model = copy.deepcopy(self.model)
                cur_loss = avg_loss
                #print(old_model.state_dict())
                print('\t\t\tfind better!', avg_loss)
                
            else:
                rand_num = np.random.rand(1)[0]
                th = math.exp( (cur_loss - avg_loss) / T)

                if rand_num < th:
                    del old_model
                    old_model = copy.deepcopy(self.model)
                    cur_loss = avg_loss	
                    print('\t\t\trand:', rand_num, 'th:', th, 'new valid loss:', cur_loss)
                else:
                    pass
        
        del self.model
        self.model = copy.deepcopy(old_model)
        del old_model
        
    def _add_random_weights(self, m):
        if isinstance(m, nn.Conv2d):
            #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            rand_num = torch.randn(m.out_channels, m.in_channels, m.kernel_size[0], m.kernel_size[1])
            #print(m.weight.data)
            #print(self.to_gpu(rand_num))
            m.weight.data = torch.add(m.weight.data, 0.001, self.to_gpu(rand_num))
        elif isinstance(m, nn.Linear):
            #n = m.in_features * m.out_features
            rand_num = torch.randn(m.in_features * m.out_features)
            m.weight.data = torch.add(m.weight.data, 0.001, self.to_gpu(rand_num))
        else:
            pass 
        
        #rand_num = torch.randn(n)    
        #torch.add(m.weight.data, 0.001, self.to_gpu(rand_num))
    
    def _get_nn_neighbor(self, nn):
        return nn.apply(self._add_random_weights)
    
    def test(self, test_df, is_transfer_learning = False, is_general_model=False, is_augment=False, data_augmentation_args={}):
        # set the model status (train\test)
        self.model.train(False)
        
        test_data_transforms = self.get_data_transforms(is_augment, is_transfer_learning, data_augmentation_args)
            
        test_dataset = IcebergDataset(
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
        
        pretrained_dict = state['state_dict']
        model_dict = self.model.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        pretrained_dict = torch.load(path)
        model_dict = self.model.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)