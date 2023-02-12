import sys
sys.path.append('./')
from data.img import IMG
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data import Sampler
from torchvision import transforms as T
from torchvision.transforms.transforms import RandomResizedCrop
from data.celeba import CelebA
from data.attr_dataset import AttributeDataset
from data.img import IMG
from functools import reduce
import math
import warnings
import torchvision.transforms.functional as vision_F


class DCLTransform:
    """Create two crops of the same image"""

    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform(x), self.transform2(x)]


class RotTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        x = self.transform(x)
        return [x, vision_F.rotate(x, 90), vision_F.rotate(x, 180), vision_F.rotate(x, 270)]


transforms = {
    "ColoredMNIST": {
        "train": T.Compose([T.ToTensor()]),
        "eval": T.Compose([T.ToTensor()])
        },
    "CorruptedCIFAR10": {
        "train": T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        "eval": T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        "train_rot": RotTransform(T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
    },
    "CelebA": {
        "train": T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
        ]),
        "eval": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    },
    "BIRD": {
        "train": T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
        ]),
        "eval": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "train_dcl": DCLTransform(
            T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
            ]),
            T.Compose([
                T.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        ),
    },
}


def get_dataset(dataset_tag, data_dir, dataset_split, transform_split):
    dataset_category = dataset_tag.split("-")[0]
    root = os.path.join(data_dir, dataset_tag)
    transform = transforms[dataset_category][transform_split]
    dataset_split = "valid" if (dataset_split == "eval") else dataset_split
    if dataset_tag == "CelebA":
        dataset = CelebA(
            root=data_dir,
            split=dataset_split,
            target_type="attr",
            transform=transform,
        )
        male_idx = (dataset.attr[:, 20]==1)
        female_idx = (dataset.attr[:, 20]==0)
        dataset.attr[female_idx, 20] = 1
        dataset.attr[male_idx, 20] = 0
        if dataset_split=='train':
            train_target_attr = dataset.attr[:, 9]
            train_bias_attr = dataset.attr[:, 20]
            dataset.data = np.concatenate((
                dataset.data[torch.where((train_target_attr==0) & (train_bias_attr==0))[0][:22880]],
                dataset.data[torch.where((train_target_attr==1) & (train_bias_attr==1))[0][:22880]],
                dataset.data[torch.where((train_target_attr==0) & (train_bias_attr==1))[0][:231]],
                dataset.data[torch.where((train_target_attr==1) & (train_bias_attr==0))[0][:231]],
            ))
            dataset.attr = torch.cat((
                dataset.attr[torch.where((train_target_attr==0) & (train_bias_attr==0))[0][:22880]],
                dataset.attr[torch.where((train_target_attr==1) & (train_bias_attr==1))[0][:22880]],
                dataset.attr[torch.where((train_target_attr==0) & (train_bias_attr==1))[0][:231]],
                dataset.attr[torch.where((train_target_attr==1) & (train_bias_attr==0))[0][:231]],
            ))
    elif dataset_tag=='BIRD':
        dataset = IMG(
            root=data_dir,
            split=dataset_split,
            transform=transform,
        )
        if dataset_split=='train':
            train_target_attr = dataset.attr[:, 0]
            train_bias_attr = dataset.attr[:, 1]
            dataset.data = np.concatenate((
                dataset.data[torch.where((train_target_attr==0) & (train_bias_attr==0))[0][:1057]],
                dataset.data[torch.where((train_target_attr==1) & (train_bias_attr==1))[0][:1057]],
                dataset.data[torch.where((train_target_attr==0) & (train_bias_attr==1))[0][:56]],
                dataset.data[torch.where((train_target_attr==1) & (train_bias_attr==0))[0][:56]],
            ))
            dataset.attr = torch.cat((
                dataset.attr[torch.where((train_target_attr==0) & (train_bias_attr==0))[0][:1057]],
                dataset.attr[torch.where((train_target_attr==1) & (train_bias_attr==1))[0][:1057]],
                dataset.attr[torch.where((train_target_attr==0) & (train_bias_attr==1))[0][:56]],
                dataset.attr[torch.where((train_target_attr==1) & (train_bias_attr==0))[0][:56]],
            ))     
    else:
        dataset = AttributeDataset(
            root=root, split=dataset_split, transform=transform
        )

    return dataset
