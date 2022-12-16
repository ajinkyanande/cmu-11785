import os
from PIL import Image

import torch
import torchvision

from params import *


class ClassificationTestDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, transforms):
        
        self.data_dir = data_dir
        self.transforms = transforms

        # sorted path list to images in test folder
        self.img_paths = list(map(lambda fname: os.path.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

    def __len__(self):
        
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        
        return self.transforms(Image.open(self.img_paths[idx]))


def get_datasets():

    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR,
                                                     transform=TRAIN_TRANSFORMS)

    val_dataset   = torchvision.datasets.ImageFolder(VAL_DIR,
                                                     transform=VAL_TRANSFORMS)
    
    test_dataset  = ClassificationTestDataset(TEST_DIR,
                                              transforms=TEST_TRANSFORMS)
    
    return train_dataset, val_dataset, test_dataset


def get_loaders(train_dataset, val_dataset, test_dataset):

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=CONFIG['batch_size'], 
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)

    val_loader   = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=CONFIG['batch_size'], 
                                               shuffle=False,
                                               num_workers=2)

    test_loader  = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=CONFIG['batch_size'],
                                               shuffle=False,
                                               drop_last=False,
                                               num_workers=2)

    # data info
    print('train images:', len(train_dataset))
    print('classification classes:', len(train_dataset.classes))
    print('feature batch shape:', next(iter(train_loader))[0].shape)

    return train_loader, val_loader, test_loader