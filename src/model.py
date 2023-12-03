""" This code is from Zhang, T., Trad, D., & Innanen, K. (2023). 
Learning to solve the elastic wave equation with Fourier neural operators. 
Geophysics, 88(3), T101-T119.

Modified by: Haipeng Li
Email: haipeng@stanford.edu

Stanford Earth imaging Project, Stanford University
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utilities3 import *


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.

        input: batchsize x channel x x_grid x y_grid (channel means the number of input channels)
        output: batchsize x channel x x_grid x y_grid

        Parameters:
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        modes1: int
            Number of Fourier modes in x direction.
        modes2: int
            Number of Fourier modes in y direction.
        """

        super(SpectralConv2d_fast, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1/(self.in_channels*self.out_channels*1000)) # here 10 means the dx and dz
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, \
                                                             self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, \
                                                             self.modes1, self.modes2, dtype=torch.cfloat))

        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, \
                                                             self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, \
                                                             self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        """ (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        """

        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """ Multiply relevant Fourier modes and return to physical space
        """

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft_x = torch.fft.rfft2(x)
        x_ft_z = torch.fft.rfft2(torch.transpose(x,2,3))

        # Multiply relevant Fourier modes
        out_ft_x = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, \
                             dtype=torch.cfloat, device=x.device)
        out_ft_z = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, \
                             dtype=torch.cfloat, device=x.device)
        out_ft_x[:, :, :self.modes1, :self.modes2]  = self.compl_mul2d(x_ft_x[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft_x[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft_x[:, :, -self.modes1:, :self.modes2], self.weights2)


        out_ft_z[:, :, :self.modes1, :self.modes2]  = self.compl_mul2d(x_ft_z[:, :, :self.modes1, :self.modes2], self.weights3)
        out_ft_z[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft_z[:, :, -self.modes1:, :self.modes2], self.weights4)

        #Return to physical space
        out_ft_x = torch.fft.irfft2(out_ft_x, s=(x.size(-2), x.size(-1)))
        out_ft_z = torch.fft.irfft2(out_ft_z, s=(x.size(-2), x.size(-1)))
        
        return out_ft_x, out_ft_z
    


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear((50+5), self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv6 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w4 = nn.Conv2d(self.width, self.width, 1)
        self.w5 = nn.Conv2d(self.width, self.width, 1)
        self.w6 = nn.Conv2d(self.width, self.width, 1)

        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)
        self.bn4 = torch.nn.BatchNorm2d(self.width)
        self.bn5 = torch.nn.BatchNorm2d(self.width)
        self.bn6 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 300)

    def forward(self, x , C1, C2, C3):
        """ The forward propagation of neural network

        Parameters:
        ----------
        x: the solution of the previous timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        C1, C2, C3: the concentration of the three species
        """

        batchsize = x.shape[0]

        size_x, size_y = x.shape[1], x.shape[2]

        C1 = torch.reshape(C1,[batchsize,1,size_x,size_y])
        C2 = torch.reshape(C2,[batchsize,1,size_x,size_y])
        C3 = torch.reshape(C3,[batchsize,1,size_x,size_y])

        C1 = torch.nn.functional.normalize(C1)
        C2 = torch.nn.functional.normalize(C2)
        C3 = torch.nn.functional.normalize(C3)

        C1 = torch.reshape(C1,[batchsize,size_x,size_y,1])
        C2 = torch.reshape(C2,[batchsize,size_x,size_y,1])
        C3 = torch.reshape(C3,[batchsize,size_x,size_y,1])

        grid = self.get_grid(batchsize, size_x, size_y, x.device)
        x = torch.cat((x, grid, C1, C2, C3), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x0, z0 = self.conv0(x)
        xw_0   = self.w0(x)
        x      = self.bn0(xw_0 + x0 + z0 )
        x = F.relu(x)
        
        x1, z1 = self.conv1(x)
        xw_1   = self.w1(x)
        x      = self.bn1(xw_1 + x1 + z1 + x)
        x = F.relu(x)
        
        x2, z2 = self.conv2(x)
        xw_2   = self.w2(x)
        x      = self.bn2(xw_2 + x2 + z2 + x)
        x = F.relu(x)
        
        
        x3, z3 = self.conv3(x)
        xw_3   = self.w3(x)
        x      = self.bn3(xw_3 + x3 + z3 + x)
        x = F.relu(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)

        return x

    def get_grid(self, batchsize, size_x, size_y, device):
        """ Generate coordinate grid scaled to interval [-1,1]
        """

        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        
        return torch.cat((gridx, gridy), dim=-1).to(device)