
from model import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#d = Discriminator().to(device)
#print(d)
g = Generator2().to(device)
print(g)

input = torch.rand(3,128).to(device)
output=g(input)
print(output.shape)
#print(output)

#input = torch.rand(3,3,256,256).to(device)
#output = d(input)
#print(output.shape)
#print(output)


print('1')