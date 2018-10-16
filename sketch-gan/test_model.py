
from model import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


#d = Discriminator()
#print(d)
g= Generator2()
print(g)

input = torch.rand(10,128)
output=g(input)
print(output.shape)
#print(output)

#input = torch.rand(10,3,256,256)
#output = d(input)
#print(output.shape)
#print(output)


print('1')