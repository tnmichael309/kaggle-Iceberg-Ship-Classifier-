import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function

class L1Penalty(Function):

    @staticmethod
    def forward(ctx, input, l1weight):
        ctx.save_for_backward(input)
        ctx.l1weight = l1weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = input.clone().sign().mul(self.l1weight)
        grad_input += grad_output
        return grad_input
        
class SWWAE(nn.Module):
    def __init__(self, l1weight=0.0, feature_extract=False):
        super(SWWAE, self).__init__()
        
        self.l1weight = l1weight
        self.feature_extract = feature_extract
        
        #Lout=floor(Lin+2*padding-kernel_size+1)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(4,32,kernel_size=6),   # batch x 16 x 70 x 70
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.1)
        )
        
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=5),   # batch x 32 x 66 x 66
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.1)
        )
        # max pool: # batch x 32 x 33 x 33
        
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=5),   # batch x 32 x 29 x 29
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.1)
        )
        
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=4),   # batch x 64 x 26 x 26
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.1)
        )
        # max pool: # batch x 64 x 13 x 13
        
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=4),   # batch x 64 x 10 x 10
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.1)
        )
        
        self.conv_layer6 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=4),   # batch x 64 x 7 x 7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.1)
        )
        
        
        self.fc1 = nn.Linear(64*7*7, 1024) 
        self.bn1 = nn.BatchNorm1d(1024, momentum=0.1) #1024 number of latent representation we want to learn
        
        self.fc2 = nn.Linear(1024, 1) 
        
        self.bn3 = nn.BatchNorm1d(1024, momentum=0.1)
        self.fc3 = nn.Linear(1024, 64*7*7)
        
        
        #Lout=Lin−1−2∗padding+kernel_size+output_padding
        self.deconv_layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4),   # batch x 64 x 10 x 10
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.deconv_layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4),   # batch x 64 x 13 x 13
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # max unpool: # batch x 64 x 26 x 26
        
        self.deconv_layer3 = nn.Sequential(
            nn.ConvTranspose2d(64,32,kernel_size=4),   # batch x 32 x 29 x 29
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.deconv_layer4 = nn.Sequential(
            nn.ConvTranspose2d(32,32,kernel_size=5),   # batch x 32 x 33 x 33
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # max unpool: # batch x 64 x 66 x 66
        
        self.deconv_layer5 = nn.Sequential(
            nn.ConvTranspose2d(32,32,kernel_size=5),   # batch x 32 x 70 x 70
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.deconv_layer6 = nn.Sequential(
            nn.ConvTranspose2d(32,4,kernel_size=6),   # batch x 32 x 75 x 75
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        
        '''
        self.conv1 = nn.Conv2d(4, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20*15*15, 2048)
        self.fc2 = nn.Linear(2048, 1)
        # Deconvolution
        self.fc3 = nn.Linear(2048, 20*15*15)
        #self.fc4 = nn.Linear(64, 20*15*15)
        self.deconv1 = nn.ConvTranspose2d(20, 10, kernel_size=6)
        self.deconv2 = nn.ConvTranspose2d(10, 4, kernel_size=6)
        '''

    def forward(self, x):
        '''
        x, indices1 = F.max_pool2d(self.conv1(x), 2, return_indices=True)
        #print(x) #128x10x35x35
        x_CONVI = F.relu(x)
        x, indices2 = F.max_pool2d(self.conv2_drop(self.conv2(x_CONVI)), 2, return_indices=True)
        #print(x) #128x20x15x15
        x_CONVII = F.relu(x)
        x = x_CONVII.view(-1, 20*15*15)
        
        x = self.fc1(x)
        x = F.relu(x)
        x_bce= nn.Sigmoid()(self.fc2(x))
        
        x = F.relu(self.fc3(x))
        x_DECONVII = x.view(self.get_size(x),20,15,15)
        x_DECONVI = F.relu(self.deconv1(F.max_unpool2d(x_DECONVII, indices2 , 2, 2)))
        output = F.relu(self.deconv2(F.max_unpool2d(x_DECONVI, indices1 , 2, 2)))
        
        '''
        
        x=self.conv_layer1(x)
        #print(x)
        x=self.conv_layer2(x)
        #print(x)
        x_conv_1, indices1 = F.max_pool2d(x, 2, return_indices=True)
        x=self.conv_layer3(x_conv_1)
        #print(x)
        x=self.conv_layer4(x)
        #print(x)
        x_conv_2, indices2 = F.max_pool2d(x, 2, return_indices=True)
        x=self.conv_layer5(x_conv_2)
        x=self.conv_layer6(x)
        
        #print(x)
        x = x.view(self.get_size(x),64*7*7)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x) #encoder output
        enc_feat = x
        
        x = L1Penalty.apply(x, self.l1weight)
        
        x_bce = self.fc2(x)
        x_bce= nn.Sigmoid()(x_bce)
        
        x = self.fc3(x) #connect to the output of the encoder
        x = x.view(self.get_size(x),64,7,7)
        
        x=self.deconv_layer1(x)
        x_deconv_2=self.deconv_layer2(x)
        x = F.max_unpool2d(x_deconv_2, indices2 , 2, 2)
        x=self.deconv_layer3(x)
        x_deconv_1=self.deconv_layer4(x)
        x = F.max_unpool2d(x_deconv_1, indices1 , 2, 2)
        x=self.deconv_layer5(x)
        decode_output=self.deconv_layer6(x)
        
        if self.feature_extract is True:
            return enc_feat
        else:  
            return x_conv_1, x_conv_2, x_deconv_2, x_deconv_1, decode_output, x_bce
    
    def get_size(self, x):
        return x.size()[0]
        
def get_swwae(l1weight=0.0, feature_extract=False):
    return SWWAE(l1weight=l1weight, feature_extract=feature_extract)
