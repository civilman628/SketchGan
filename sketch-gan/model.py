import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    '''Residual block with instance normalization'''
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)
            )

    def forward(self,x):
        return x + self.layers(x)

class DownSample(nn.Module):
    def __init__(self):
        super(DownSample,self).__init__()

    def forward(self,x):
        x = F.interpolate(x,scale_factor=0.5)
        return x

class UpSample(nn.Module):
    def __init__(self):
        super(UpSample,self).__init__()

    def forward(self,x):
        x = F.interpolate(x,scale_factor=2)
        return x

class ResidualBlock2(nn.Module):
    '''Residual block'''
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding=1,bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1,bias=False)
        nn.init.orthogonal(self.conv1.weight.data, 1.)
        nn.init.orthogonal(self.conv2.weight.data, 1.)

        self.layers=nn.Sequential(
            nn.ReLU(inplace=True),
            self.conv1,
            nn.ReLU(inplace=True),
            self.conv2,
            )

    def forward(self,x):
        return x + self.layers(x)

class GlobalSumPooling(nn.Module):
    '''Global Sum Pooling'''
    def __init__(self):
        super(GlobalSumPooling, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0),x.size(1),-1)
        x = torch.sum(x,dim=2)
        return x

class ResidualBlockUp(nn.Module):
    '''Residual block with instance normalization'''
    def __init__(self, in_channels, out_channels,ch=64):
        super(ResidualBlockUp,self).__init__()
        self.conv1 = nn.Conv2d(in_channels*ch, out_channels*ch, 3, 1, padding=1,bias=False)
        self.conv2 = nn.Conv2d(out_channels*ch, out_channels*ch, 3, 1, padding=1,bias=False)
        self.identity = nn.Conv2d(in_channels*ch, out_channels*ch, 1, 1)
        nn.init.orthogonal(self.conv1.weight.data, 1.)
        nn.init.orthogonal(self.conv2.weight.data, 1.)
        nn.init.orthogonal(self.identity.weight.data, 1.)
        #nn.init.orthogonal(self.conv1.weight.data)

        #self.model =[]
        #model.append(nn.BatchNorm2d(in_channels))
        #model.append(nn.ReLU)
        #model.append
        self.shortcut= nn.Sequential(
            nn.BatchNorm2d(in_channels*ch),
            nn.ReLU(inplace=True),
            UpSample(),
            self.identity,
            )
        
        self.model=nn.Sequential(
            nn.BatchNorm2d(in_channels*ch),
            nn.ReLU(inplace=True),
            self.conv1,
            nn.BatchNorm2d(out_channels*ch),
            nn.ReLU(inplace=True),
            UpSample(),
            self.conv2
            )

    def forward(self, x):
        right = self.shortcut(x)
        x = self.model(x) 
        return x + right

class ResidualBlockDown(nn.Module):
    '''Residual block with instance normalization'''
    def __init__(self, in_channels, out_channels,ch=64):
        super(ResidualBlockDown,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels*ch, 3, 1, padding=1)
        self.identity = nn.Conv2d(in_channels, out_channels*ch, 1, 1)
        nn.init.orthogonal(self.conv1.weight.data, 1.)
        nn.init.orthogonal(self.conv2.weight.data, 1.)
        nn.init.orthogonal(self.identity.weight.data, 1.)

        #self.model =[]
        #model.append(nn.BatchNorm2d(in_channels))
        #model.append(nn.ReLU)
        #model.append

        self.shortcut= nn.Sequential(
            nn.ReLU(inplace=True),
            self.identity,
            DownSample()
            )
        
        self.model=nn.Sequential(
            #nn.BatchNorm2(in_channels),
            nn.ReLU(inplace=True),
            self.conv1,
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DownSample(),
            self.conv2
            )

    def forward(self, x):
        left = self.model(x) 
        right = self.shortcut(x)
        return left + right

class Generator2(nn.Module):
    def __init__(self,n_dim=128,ch=64):
        super(Generator2,self).__init__()
        self.ch=ch
        self.denselayer = nn.Linear(n_dim,4*4*16*self.ch)
        self.final = nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1,bias=False)

        nn.init.orthogonal(self.denselayer.weight.data, 1.)
        nn.init.orthogonal(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResidualBlockUp(16,16),
            ResidualBlockUp(16,8),
            ResidualBlockUp(8,8),
            ResidualBlockUp(8,4),
            ResidualBlockUp(4,2),
            ResidualBlockUp(2,1),
            nn.BatchNorm2d(self.ch),
            nn.ReLU(inplace=True),
            self.final,
            nn.Tanh()
            )

    def forward(self,x):
        x=self.denselayer(x)
        x=x.view(-1, 16*self.ch, 4,4)
        x=self.model(x)

        return x


class Discriminator(nn.Module):
    "Discriminator network."
    def __init__(self, ch=64):
        super(Discriminator,self).__init__()

        self.model = nn.Sequential(
            ResidualBlockDown(3,1),
            ResidualBlockDown(1*ch,2),
            ResidualBlockDown(2*ch,4),
            ResidualBlockDown(4*ch,8),
            ResidualBlockDown(8*ch,8),
            ResidualBlockDown(8*ch,16),
            ResidualBlock2(16*ch,16*ch),
            nn.ReLU(inplace=True),
            GlobalSumPooling(),
            nn.Linear(16*ch,1)
            )


    def forward(self, x):
        x=self.model(x)
        return x
