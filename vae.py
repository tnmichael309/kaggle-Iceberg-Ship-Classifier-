import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision
import math


# VAE model        
class VAE(nn.Module):
    def __init__(self, image_size=5625, h_dim=400, z_dim=20):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, image_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.image_size = image_size

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.image_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        