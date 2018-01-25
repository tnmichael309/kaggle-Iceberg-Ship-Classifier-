import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(4,16,6,padding=1),   # batch x 32 x 75 x 75
                        nn.ReLU(),
                        nn.BatchNorm2d(16),
                        nn.Conv2d(16,16,6,padding=1),   # batch x 32 x 75 x 75
                        nn.ReLU(),
                        nn.Dropout2d(p=0.2),
                        nn.BatchNorm2d(16),
                        nn.Conv2d(16,32,6,padding=1),  # batch x 64 x 75 x 75
                        nn.ReLU(),
                        nn.Dropout2d(p=0.2),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,32,6,padding=1),  # batch x 64 x 75 x 75
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.MaxPool2d(2,2),   # batch x 64 x 37 x 37
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(32,64,6,padding=1),  # batch x 128 x 37 x 37
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64,64,6,padding=1),  # batch x 128 x 37 x 37
                        nn.ReLU(),
                        nn.Dropout2d(p=0.2),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(64,128,6,padding=1),  # batch x 256 x 18 x 18
                        nn.ReLU(),
        )
        self.fc = nn.Linear(128*9*9, 1)
        self.prob = nn.Sigmoid()
        
    def forward(self, x, other_features):
        # other_features: ignore
        
        out = self.layer1(x)
        #print(out)
        out = self.layer2(out)
        #print(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        
        return self.prob(out)
        
def get_simple_cnn():
    return SimpleCNN()