import torch
import torchvision as tv
import numpy as np

class IcebergDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __getitem__(self, index):
        row = self.df.iloc[index]

        imgs = np.array(row['norm_band_mixed']).astype(np.uint8)
        
        if self.transform is not None:
            imgs = self.transform(imgs)
        
        other_features = np.array([row['inc_angle']]).astype(np.float32)
        other_features = torch.from_numpy(other_features)
        target = np.array([row['is_iceberg']]).astype(np.float32)
        target = torch.from_numpy(target)
        
        #sample = {'img': imgs, 'other_features': other_features, 'target': target}
        #print(imgs, other_features, target)
        
        return (imgs, other_features, target)

    def __len__(self):
        n, _ = self.df.shape
        return n

