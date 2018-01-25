import torch
import torchvision as tv
import numpy as np
import random


class IcebergDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        #self.features = 'inc_angle'
        self.features = ['inc_angle']
        for x in range(1000):
            pass
            #self.features.append('feature_' + str(x))
        
    def __getitem__(self, index):
        row = self.df.iloc[index]

        imgs = np.array(row['band_mixed']).astype(np.float)
        imgs *= 255.0
        imgs = imgs.astype(np.uint8)
        #print(imgs)
        
        if self.transform is not None:
            imgs = self.transform(imgs)
        
        other_features = np.array([row[self.features]]).astype(np.float32)
        #print(other_features.shape)
        other_features = np.squeeze(other_features, axis=0)
        other_features = torch.from_numpy(other_features)
        #print(other_features.shape)
        
        target = np.array([row['is_iceberg']]).astype(np.float32)
        target = torch.from_numpy(target)
        
        #sample = {'img': imgs, 'other_features': other_features, 'target': target}
        #print(imgs, other_features, target)
        
        return (imgs, other_features, target)

    def __len__(self):
        n, _ = self.df.shape
        return n

class ContrastDataset(torch.utils.data.Dataset):
    # df1 is the reference dataframe for data length
    def __init__(self, df1, df2, transform=None, random_pair=False):
        self.df1 = df1
        self.df2 = df2
        self.transform = transform
        self.random_pair = random_pair
        
    def __getitem__(self, index):
        row1 = self.df1.iloc[index]
        
        if self.random_pair is True:
            row2 = self.df2.iloc[random.randrange(0, self.df2.shape[0])]
        else:
            row2 = self.df2.iloc[index]
            
        def get_transform_img(row):
            img = np.array(row['band_mixed']).astype(np.float)
            img *= 255.0
            img = img.astype(np.uint8)
            
            return img
            
        img1 = get_transform_img(row1)
        img2 = get_transform_img(row2)
        #img = np.concatenate([img1, img2], axis=2)
        # print(img.shape)
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        target = np.array([row1['is_iceberg'] == row2['is_iceberg']]).astype(np.float32)
        target = torch.from_numpy(target)
        
        return (img1, img2, target)

    def __len__(self):
        n, _ = self.df1.shape
        return n
        
class PLIcebergDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        #self.features = 'inc_angle'
        self.features = ['inc_angle']
        for x in range(1000):
            pass
            #self.features.append('feature_' + str(x))
        
    def __getitem__(self, index):
        row = self.df.iloc[index]

        imgs = np.array(row['band_mixed']).astype(np.float)
        imgs *= 255.0
        imgs = imgs.astype(np.uint8)
        
        if self.transform is not None:
            imgs = self.transform(imgs)
        
        other_features = np.array([row[self.features]]).astype(np.float32)
        #print(other_features.shape)
        other_features = np.squeeze(other_features, axis=0)
        other_features = torch.from_numpy(other_features)
        #print(other_features.shape)
        
        target = np.array([row['is_iceberg']]).astype(np.float32)
        target = torch.from_numpy(target)
        
        is_pseudo = np.array([row['is_pseudo']]).astype(np.float32)
        is_pseudo = torch.from_numpy(is_pseudo)
        #sample = {'img': imgs, 'other_features': other_features, 'target': target}
        #print(imgs, other_features, target)
        #print(other_features, target, is_pseudo)
        
        return (imgs, other_features, target, is_pseudo)

    def __len__(self):
        n, _ = self.df.shape
        return n
        
class AdaptedIcebergDataset(torch.utils.data.Dataset):
    # meta_features = ['is_iceberg'] for inc_angle training\na value prediction
    # meta_featurers = ['inc_angle', 's1', 's2']
    def __init__(self, df, meta_features, target='is_iceberg', transform=None):
        self.df = df
        self.transform = transform
        self.features = meta_features
        self.target = target 
        for x in range(1000):
            pass
            #self.features.append('feature_' + str(x))
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        

        imgs = np.array(row['band_mixed']).astype(np.float)
        imgs *= 255.0
        imgs = imgs.astype(np.uint8)
        
        if self.transform is not None:
            imgs = self.transform(imgs)
        
        other_features = np.array([row[self.features]]).astype(np.float32)
        #print(other_features.shape)
        other_features = np.squeeze(other_features, axis=0)
        other_features = torch.from_numpy(other_features)
        #print(other_features.shape)
        
        target = np.array([row[self.target]]).astype(np.float32)
        target = torch.from_numpy(target)
        
        #sample = {'img': imgs, 'other_features': other_features, 'target': target}
        #print(imgs, other_features, target)
        
        return (imgs, other_features, target)

    def __len__(self):
        n, _ = self.df.shape
        return n

class VAEIcebergDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __getitem__(self, index):
        row = self.df.iloc[index]

        imgs = np.array(row['band']).astype(np.float)
        imgs *= 255.0
        imgs = imgs.astype(np.uint8)
        #imgs = imgs.reshape((imgs.shape[0], imgs.shape[1], 1))
        #print('test', imgs.shape)
        
        if self.transform is not None:
            imgs = self.transform(imgs)
        
        return imgs

    def __len__(self):
        n, _ = self.df.shape
        return n