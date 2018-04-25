import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import *
from skimage import io, transform
import scipy.ndimage as sci
plt.ion()



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(70),# translate=(15,15)),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.RandomRotation(70),
        #transforms.RandomAffine(130, translate=(15,15)),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor()
    ])
}

        
data_dir = "./"

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
batch_size=100,shuffle=True, num_workers=16)
              for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
use_gpu = torch.cuda.is_available()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    #inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
if(__name__ == "__main__"):
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    #print(repr(inputs), repr(classes))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out)






