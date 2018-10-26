import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spectral import SpectralNorm

class Nonlocal(nn.Module):
    def __init__(self,in_channels):
        ''' https://zhuanlan.zhihu.com/p/33345791 '''
        super(Nonlocal,self).__init__()
        self.inter_channels=in_channels//2

        self.theta = nn.Conv2d(in_channels,self.inter_channels,1,1)
        self.phi = nn.Conv2d(in_channels,self.inter_channels,1,1)
        self.g = nn.Conv2d(in_channels,self.inter_channels,1,1)

        self.recovery = nn.Conv2d(self.inter_channels,in_channels,1,1)
        #nn.Sequential(
            
            #nn.BatchNorm2d(in_channels)
            #)


    def forward(self, x):
        batchsize = x.size(0)
        print ('x: ',x.size())
        theta_x = self.theta(x)
        theta_x = theta_x.view(batchsize,-1,self.inter_channels)
        phi_x = self.phi(x)
        phi_x = phi_x.view(batchsize,-1,self.inter_channels)
        phi_x = phi_x.permute(0,2,1)
        f = torch.matmul(theta_x,phi_x)
        #f = F.softmax(f)
        N = f.size(-1)
        f = f / N
        print ('f: ',f.size())
        g_x = self.g(x)
        g_x = g_x.view(batchsize,-1,self.inter_channels)
        y = torch.matmul(f,g_x)
        print("y: ",y.size())
        y =  y.view(batchsize,x.size(1),x.size(2),self.inter_channels)
        print("y: ",y.size())
        y = self.recovery(y)
        print("y: ",y.size(),' x: ', x.size())
        z = x+y

        return z


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
        nn.init.orthogonal_(self.conv1.weight.data, 1.)
        nn.init.orthogonal_(self.conv2.weight.data, 1.)

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
        nn.init.orthogonal_(self.conv1.weight.data, 1.)
        nn.init.orthogonal_(self.conv2.weight.data, 1.)
        nn.init.orthogonal_(self.identity.weight.data, 1.)
        #nn.init.orthogonal(self.conv1.weight.data)

        #self.model =[]
        #model.append(nn.BatchNorm2d(in_channels))
        #model.append(nn.ReLU)
        #model.append
        self.shortcut= nn.Sequential(
            nn.BatchNorm2d(in_channels*ch),
            nn.ReLU(inplace=True),
            UpSample(),
            SpectralNorm(self.identity),
            )
        
        self.model=nn.Sequential(
            nn.BatchNorm2d(in_channels*ch),
            nn.ReLU(inplace=True),
            SpectralNorm(self.conv1),
            nn.BatchNorm2d(out_channels*ch),
            nn.ReLU(inplace=True),
            UpSample(),
            SpectralNorm(self.conv2)
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
        nn.init.orthogonal_(self.conv1.weight.data, 1.)
        nn.init.orthogonal_(self.conv2.weight.data, 1.)
        nn.init.orthogonal_(self.identity.weight.data, 1.)

        #self.model =[]
        #model.append(nn.BatchNorm2d(in_channels))
        #model.append(nn.ReLU)
        #model.append

        self.shortcut= nn.Sequential(
            nn.ReLU(inplace=True),
            SpectralNorm(self.identity),
            DownSample()
            )
        
        self.model=nn.Sequential(
            #nn.BatchNorm2(in_channels),
            nn.ReLU(inplace=True),
            SpectralNorm(self.conv1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DownSample(),
            SpectralNorm(self.conv2)
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

        nn.init.orthogonal_(self.denselayer.weight.data, 1.)
        nn.init.orthogonal_(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResidualBlockUp(16,16),
            ResidualBlockUp(16,8),
            ResidualBlockUp(8,8),
            ResidualBlockUp(8,4),
            ResidualBlockUp(4,2),
            Nonlocal(2*ch),
            ResidualBlockUp(2,1),
            nn.BatchNorm2d(self.ch),
            nn.ReLU(inplace=True),
            SpectralNorm(self.final),
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
            Nonlocal(2*ch),
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
