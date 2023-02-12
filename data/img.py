import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class IMG(Dataset):
    def __init__(self, root, split, transform=None):
        super(IMG, self).__init__()
        if split=='train':
            data_path = os.path.join(root, "%s.txt"%(split))
        else:
            data_path = os.path.join(root, "%s.txt"%(split))
        self.data = []
        self.attr = []
        with open(data_path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                tmp = l.strip().split('###')
                name = os.path.join(root, 'img', tmp[0])
                self.data.append(name)
                self.attr.append([int(tmp[1]), int(tmp[2])])
        self.attr = torch.LongTensor(self.attr)
        self.transform = transform
        self.data = np.array(self.data)

    def __len__(self):
        return self.attr.size(0)

    def __getitem__(self, index):
        image = Image.open(
            self.data[index]
        ).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return index, image, self.attr[index]
