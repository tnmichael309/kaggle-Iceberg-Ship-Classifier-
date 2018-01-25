import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4,16,6,padding=1),   # batch x 32 x 75 x 75
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16,16,6,padding=1),   # batch x 32 x 75 x 75
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            
            nn.Conv2d(16,32,6,padding=1),  # batch x 64 x 75 x 75
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            
            nn.Conv2d(32,32,6,padding=1),  # batch x 64 x 75 x 75
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2,return_indices=True),   # batch x 64 x 37 x 37
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,6,padding=1),  # batch x 128 x 37 x 37
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64,64,6,padding=1),  # batch x 128 x 37 x 37
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,return_indices=True),
        )
        
        #self.fc = nn.Linear(128*11*11, 512)
        
    def forward(self,x):
        out, max_indices_1 = self.layer1(x)
        #print(out) #64x32x31x31
        out, max_indices_2 = self.layer2(out)
        #print(out) #64x64x12x12
        out = out.view(x.size(0), -1)
        #out = self.fc(out)
        
        return out, max_indices_1, max_indices_2
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        
        #self.fc = nn.Linear(512, 128*11*11)
        
        #self.max_pool = nn.MaxUnpool2d(2,2) # 11->22
        # 22 -> 31
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(64,64,6,padding=1),   # batch x 128 x 37 x 37
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64,32,7,padding=1),    # batch x 64 x 37 x 37
            nn.BatchNorm2d(32),
            nn.ReLU(), 
        )
        #self.max_pool = nn.MaxUnpool2d(2,2)                
                        
        self.layer2 = nn.Sequential(
            

            nn.ConvTranspose2d(32,32,6,padding=1),     # batch x 64 x 37 x 37
            nn.BatchNorm2d(32),
            nn.ReLU(),
                
            nn.ConvTranspose2d(32,16,6,padding=1),     # batch x 32 x 37 x 37
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            
            nn.ConvTranspose2d(16,16,6,padding=1),     # batch x 32 x 37 x 37
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            
            nn.ConvTranspose2d(16,4,7,padding=1),    # batch x 1 x 75 x 75
            nn.BatchNorm2d(4),
            nn.ReLU()             
        )
        
        self.max_unpool = nn.MaxUnpool2d(2,2)
        self.prob = nn.Sigmoid()
        
    def forward(self, x, max_indices_1, max_indices_2):
        #print(self.batch_size)
        #x = self.fc(x)
        
        out = x.view(x.size(0),64,12,12)
        #print(out) bsx64x12x12
        out = self.max_unpool(out, max_indices_2)
        #print(out) #bsx64x24x24
        out = self.layer1(out)
        #print(out) #bsx64x31x31
        
        out = self.max_unpool(out, max_indices_1)
        #print(out) #bsx64x62x62
        out = self.layer2(out) 
        #print(out) #bsx64x75x75
        
        return self.prob(out)
    
class Denoise_AE(nn.Module):
    def __init__(self):
        super(Denoise_AE,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self,x):
        x, max_indices_1, max_indices_2 = self.encoder(x)
        return self.decoder(x, max_indices_1, max_indices_2)

class Simple_DAE(nn.Module):
    def __init__(self, image_size=5625, h_dim=2048, z_dim=256):
        super(Simple_DAE, self).__init__()

        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, image_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.image_size = image_size

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc2(h1)


    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        x = x.view(x.size(0),-1)
        #print(x)
        enc = self.encode(x)
        dec = self.decode(enc)
        return dec.view(dec.size(0), 1, 75, 75)
        
def DAE():
    #return Simple_DAE()
    return Denoise_AE()