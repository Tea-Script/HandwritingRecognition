
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

from common_functions import *



#image processing
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'label': label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        image = image.transpose((2, 0, 1))
        return (torch.from_numpy(image),
                torch.from_numpy(label))


data_transforms = {
    'train': transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        ToTensor()
    ]),
    'test': transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        ToTensor()
    ])
}


def get_indices(root_dir, datafolder):
    sizes = []
    path = os.path.join(root_dir, datafolder)
    for batch in sorted(os.listdir(path)):
        path2 = os.path.join(path, batch)
        if(os.path.isdir(path2)):
            x = subprocess.check_output(['ls','-l', '{}'.format(path2)])
            x = len(x.splitlines()) - 1
            sizes.append(x)
    cum_sizes = [0] * len(sizes)
    for i in range(len(sizes)):
        for j in range(i+1):
            cum_sizes[i] += sizes[j]
    indices = [0]*len(sizes)
    for i in range(len(indices)):
        if(i - 1 < 0):
            indices[i] = list(range(cum_sizes[i]))
        else:
            indices[i] = list(range(cum_sizes[i-1],cum_sizes[i]))
    return indices
    
    
get_indices("./", "train")

r = list(range(33, 123)) #keyboard values of ascii table
blacklist = [92,94,95,35,36,37,38, 39]
r = [chr(x) for x in r if x not in blacklist] #remove special characters and escape characters
class_names = r + supported_characters + [' ', "#", "$", "&"]
#print(class_names)




class BatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, folder, batch_size=0, drop_last=False):
        '''if not isinstance(sampler, torch.utils.data.sampler.SequentialSampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.SequentialSampler, but got sampler={}"
                             .format(sampler))
        
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        '''
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.currentbatch = 0
        self.batches = get_indices("./", folder)
    def __iter__(self):
        #if(self.currentbatch < len(self.batches)):
        #    yield self.batches[self.currentbatch]
        #self.currentbatch += 1
        return iter(self.batches)
    def __len__(self):
        return len(self.batches)

class SymbDataset(Dataset):
    """Dataset Class For CNN"""

    def __init__(self, root_dir, classnames=None, transform=None):
        """
        Args:
            root_dir (string): Directory containing all of the images and tex files.
            classnames (list): List of all of the possible classes
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.len = None #calculate length only once
        self.classnames = classnames
        self.docs = []
        for file in os.listdir(root_dir):
            #print(file)
            if file.endswith(".tex"):
                path = os.path.join(root_dir, file)
                with open(path, 'r') as f:
                    self.docs.append( (  file , simplify(f.read(), 0) ) ) #tup containing file, expected result values pairs
        self.root_dir = root_dir
        self.transform = transform
        #print(self.docs)

    def __len__(self): #returns number of images
        path = self.root_dir
        tot = get_indices("./", path)[-1][-1]
        self.len = tot
        return tot

    def len2(self): #returns number of batches
        return len(self.docs)
    def get_idx(self, idx):
        #finds the batch number given an index of all the images
        batch = 0
        cum = 0
        l=0
        while(idx > 0):
            path = os.path.join(self.root_dir, str(batch))
            l = len(os.listdir(path))
            if(idx >= l): 
                batch += 1
                idx -= l
                cum +=l
            else: break

        self.idx1 = batch
        self.idx2 = idx
            
    def __getitem__(self, idx):
        self.get_idx(idx)
        idx1 = self.idx1
        idx2 = self.idx2
        imglabel = self.docs[idx1][1] #label with file contents
        #print(imglabel)
        imglabel = np.array([self.classnames.index(classname) for classname in imglabel]) #array with the indices for each class in classnames
        #print(imglabel)


        imgdir = os.path.join(self.root_dir, self.docs[idx1][0].strip(".tex"))
        img = None
        l = idx2
        
        for file in sorted(os.listdir(imgdir)):
            file = os.path.join(imgdir, file)
            print(file)
            if(l == 0):
                img = sci.imread(file, mode="RGB")
                if(img is None):
                    return __getitem__(idx+1)
                                 
            l -= 1
        #sample = np.array((img , imglabel))
        #print(img.shape, imglabel.shape)
        sample = {'image': img, 'label': imglabel}
        if self.transform:
            sample = self.transform(sample)

        return sample
        
data_dir = "./"

image_datasets = {x: SymbDataset(os.path.join(data_dir, x), classnames = class_names ,
                                          transform = data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_sampler = BatchSampler("./", x),
                                              num_workers=0) 
              for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

use_gpu = torch.cuda.is_available()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
#print(repr(inputs), repr(classes))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out)






