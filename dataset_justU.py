import os
import numpy as np

import torch
import torch.nn as nn

##
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir_data, transform=None):
        self.dir_data = dir_data
        self.transform = transform

        lst_data = os.listdir(self.dir_data)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):

        label = np.load(os.path.join(self.dir_data, self.lst_label[index]))        
        input = np.load(os.path.join(self.dir_data, self.lst_input[index]))
        name = self.lst_input[index][6:14]
        input = input[:,:,np.newaxis]

        data = {'name': name,'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)
            
        return data

class ToTensor(object):
    def __call__(self, data):
        name, label, input = data['name'], data['label'], data['input']

        label = label.transpose.astype(np.int64)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'name': name, 'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, data):
        name, label, input = data['name'], data['label'], data['input']

        input = (input - self.min) / (self.max - self.min)

        data = {'name': name, 'label': label, 'input': input}

        return data