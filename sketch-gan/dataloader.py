import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os


class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_dresses(self):
        transforms = self.transform(False, False, True, False)
        dataset = ImageFolder(self.path+'/dresses', transform=transforms)
        return dataset

    def load_tops(self):
        transforms = self.transform(False, True, True, False)
        dataset = dsets.ImageFolder(self.path+'/tops', transform=transforms)
        return dataset


    def loader(self):
        '''
        if self.dataset == 'dresses':
            dataset = self.load_dresses()
        elif self.dataset == 'tops':
            dataset = self.load_tops()
        '''
        transforms = self.transform(False, False, True, False)
        #print (os.path.abspath('../../data/dresses'))
        dataset = ImageFolder( self.path, transform=transforms)

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=2)
                                              
        return loader
