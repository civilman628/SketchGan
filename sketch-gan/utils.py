import os
import torch
from torch.autograd import Variable


def make_folder(path, version):
        if not os.path.exists(os.path.join(path, version)):
            os.makedirs(os.path.join(path, version))


def cudatensor(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.tensor(x, requires_grad=grad)

def var2tensor(x):
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

