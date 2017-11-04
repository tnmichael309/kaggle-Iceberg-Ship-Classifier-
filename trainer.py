import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from data_set_api import IcebergDataset
from torch.utils.data.dataloader import default_collate
import torchvision as tv
import numpy as np
from random_affine_api import RandomAffineTransform
import math

class Trainer:
    def __init__(self, 
        model,
        optimizer = None,
        epochs=300,
        milestones=[60, 160, 260],
        gamma=0.1,
        batch_size=256, 
        use_cuda=False, 
        gpu_idx=0):
        
        self.use_cuda = use_cuda
        self.gpu_idx = gpu_idx
        
        if self.use_cuda:
            torch.cuda.set_device(gpu_idx)
            print("gpu:",gpu_idx," available:", torch.cuda.is_available())
            
            with torch.cuda.device(gpu_idx):
                self.model = self.to_gpu(model)
        else:
            self.model = self.from_gpu(model)
        
        if self.model is None:
            raise ValueError('Cannot initialize the model')
            
        print("Model in use:")
        print(model)
        
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.set_optimizer(optimizer)
        self.set_scheduler(milestones, gamma)
        self.criterion = torch.nn.BCELoss() #BCEWithLogitsLoss()
 
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

    def train(self, train_df, valid_df = None, is_transfer_learning = False):

        affine_transform = RandomAffineTransform(rotation_range=(.0,30.*math.pi/180.))

        # load data
        # what transformations should be done with our images
            
        if is_transfer_learning is True:
            data_transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.Resize((197,197)),
                tv.transforms.RandomVerticalFlip(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.Lambda(lambda img: tv.transforms.ToTensor()(affine_transform(img)))
                #tv.transforms.Lambda(lambda imgs: torch.stack([tv.transforms.ToTensor()(affine_transform(img)) for img in imgs]))
                #tv.transforms.ToTensor(),
            ])
        else:
            data_transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomVerticalFlip(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.Lambda(lambda img: tv.transforms.ToTensor()(affine_transform(img)))
                #tv.transforms.Lambda(lambda imgs: torch.stack([tv.transforms.ToTensor()(affine_transform(img)) for img in imgs]))
                #tv.transforms.ToTensor(),
            ])
        
        # initialize our dataset at first
        train_dataset = IcebergDataset(
            df=train_df,
            transform=data_transforms
        )

        # initialize data loader with required number of workers and other params
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   collate_fn=default_collate)
        if valid_df is not None:
            if is_transfer_learning is True:
                data_transforms = tv.transforms.Compose([
                    tv.transforms.ToPILImage(), # to be in range [0,1]
                    tv.transforms.Resize((197,197)),
                    tv.transforms.ToTensor(),
                ])
            else:
                data_transforms = tv.transforms.Compose([
                    tv.transforms.ToPILImage(), # to be in range [0,1]
                    tv.transforms.ToTensor(),
                ])
                
            valid_dataset = IcebergDataset(
                df=valid_df,
                transform=data_transforms
            )
            
            valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   collate_fn=default_collate)
                                                   
        for epoch in range(self.epochs):
            self.scheduler.step()
            is_show = (epoch+1) % 10 == 0
            
            if is_show is True:
                print('epoch=', epoch+1, end=': ')
                
            self.model.train(True)
            self.train_epoch(train_data_loader, is_show)
            
            if valid_df is not None and is_show is True:
                self.model.train(False)
                self.valid_epoch(valid_data_loader)

    def train_epoch(self, data_loader, is_show=False):
        # set the model status (train\test)
        
        train_loss = 0
        correct = 0
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
            preds_np = outputs.data.cpu().numpy()
            labels_np = labels.data.cpu().numpy()
            preds_np = np.array([1.0 if p >= 0.5 else 0.0 for p in preds_np])
            labels_np = labels_np.reshape((labels_np.shape[0],))
            correct += (preds_np == labels_np).sum()
            counter += 1
            
        if is_show:
            train_loss /= counter
            print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                train_loss, correct, len(data_loader.dataset),
                100. * correct / len(data_loader.dataset))) 
                
    def valid_epoch(self, data_loader):
        # set the model status (train\test)
        
        valid_loss = 0
        correct = 0
        counter = 0
        for imgs, other_features, labels in data_loader:
            imgs = Variable(self.to_gpu(imgs), volatile=True)
            other_features = Variable(self.to_gpu(other_features), volatile=True)
            labels = Variable(self.to_gpu(labels), volatile=True)

            outputs = self.model(imgs, other_features)
            loss = self.criterion(outputs, labels)
            
            valid_loss += loss.data[0]
            preds_np = outputs.data.cpu().numpy()
            labels_np = labels.data.cpu().numpy()
            preds_np = np.array([1.0 if p >= 0.5 else 0.0 for p in preds_np])
            labels_np = labels_np.reshape((labels_np.shape[0],))
            correct += (preds_np == labels_np).sum()
            counter += 1
            
        valid_loss /= counter
        print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            valid_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset))) 
            
    def test(self, test_df, is_transfer_learning = False, is_general_model=False):
        # set the model status (train\test)
        self.model.train(False)
        
        if is_transfer_learning is True:
            data_transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(), # to be in range [0,1]
                tv.transforms.Resize((197,197)),
                tv.transforms.ToTensor(),
            ])
        else:
            data_transforms = tv.transforms.Compose([
                tv.transforms.ToPILImage(), # to be in range [0,1]
                tv.transforms.ToTensor(),
            ])
            
        test_dataset = IcebergDataset(
            df=test_df,
            transform=data_transforms
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
            print(pred_np.shape)
            
        predictions = np.concatenate(predictions, axis=0)
                
        return predictions
    
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))