
from model import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


d = Discriminator()
print(d)

input = torch.rand(10,3,256,256)
output = d(input)


print('1')