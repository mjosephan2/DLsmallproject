import numpy as np
import time
from collections import OrderedDict
from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models, transforms
from dataloader import Lung_Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import os
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

# AlexNet
class AlexNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, 11, 1, 1), #in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(3), #kernel_size
            nn.ReLU(inplace = True),
            nn.Conv2d(96, 256, 5, padding = 0),
            nn.MaxPool2d(3),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 384, 3, padding = 0),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 384, 3, padding = 0),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, 3, padding = 0),
            nn.MaxPool2d(3),
            nn.ReLU(inplace = True)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*2*2 , 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x

class Convolutional(Module):   
    def __init__(self, output_size):
        super(Convolutional, self).__init__()

        self.cnn_layers = Sequential(
            self.conv_bn(1, 8, kernel_size=3, stride=1),
            self.conv_bn(8, 16, kernel_size=3, stride=1),
            self.conv_bn(16, 32, kernel_size=3, stride=1),
            self.conv_bn(32, 64, kernel_size=3, stride=1),
            self.conv_bn(64, 128, kernel_size=3, stride=1),
            self.conv_bn(128, 256, kernel_size=3, stride=1),
        )
        self.classifier = NN_Classifier(input_size=256*2*2, output_size=output_size, hidden_layers=[512])
    
    def conv_bn(self, in_channels, out_channels, kernel_size=(3,3), *args, **kwargs):
        return nn.Sequential(
            Conv2dAuto(in_channels, out_channels, kernel_size=(3,3), *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activation(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetBlock(ResidualBlock):
    """
    Basic ResNet block composed by two layers of conv/batchnorm/activation
    """
    def __init__(self, in_channels, out_channels, hidden_channels=None, downsampling=1, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        if hidden_channels:
            self.hidden_channels = hidden_channels
        else:
            self.hidden_channels = self.out_channels
        self.downsampling=downsampling   
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.out_channels)) if self.should_apply_shortcut else None
        
        self.blocks = nn.Sequential(
            self.conv_bn(self.in_channels, self.hidden_channels, bias=False, stride=self.downsampling, *args, **kwargs),
            self.activation,
            self.conv_bn(self.hidden_channels, self.out_channels, bias=False, *args, **kwargs),
        )
    def conv_bn(self, in_channels, out_channels, kernel_size=(3,3), *args, **kwargs):
        return nn.Sequential(Conv2dAuto(in_channels, out_channels, kernel_size=(3,3), *args, **kwargs), nn.BatchNorm2d(out_channels))

class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, block=ResNetBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, downsampling=downsampling, *args, **kwargs),
            *[block(out_channels, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class NN_Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)


class LungResNetModel(nn.Module):
    def __init__(self, in_channels=1, n_classes=3, block_sizes=[64,128,256], depths=[2,2,2],block=ResNetBlock, activation=nn.ReLU(),
                *args, **kwargs):
        super().__init__()
        self.block_sizes = block_sizes
        self.depths = depths
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.block_sizes[0], kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(self.block_sizes[0]),
            activation
        )
        
        self.features = nn.ModuleList([ 
                ResNetLayer(self.block_sizes[0],
                            self.block_sizes[0],
                            n=self.depths[0],
                            block=block,*args, **kwargs),
                *[ResNetLayer(in_channels, 
                              out_channels,
                              n=n,
                              block=block, *args, **kwargs) 
                  for in_channels, out_channels, n in zip(self.block_sizes,self.block_sizes[1:], depths[1:])]       
        ])
        
        self.in_features = self.features[-1].blocks[-1].out_channels
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = NN_Classifier(self.in_features, n_classes, [256])
        return
    
    def forward(self,x):
        x = self.gate(x)
        for layer in self.features:
            x = layer(x)
        x = self.pooling(x)
        x = x.view(-1, self.in_features)
        x = self.classifier(x)
        return x