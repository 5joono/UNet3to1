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

        index_prev = index-1
        if self.lst_input[index][10:14] == '0000':
            index_prev = index
        index_next = index+1
        if index_next == len(self.lst_label) or self.lst_input[index_next][10:14] == '0000':
            index_next = index   

        label = np.load(os.path.join(self.dir_data, self.lst_label[index]))        
        input_prev = np.load(os.path.join(self.dir_data, self.lst_input[index_prev]))
        input = np.load(os.path.join(self.dir_data, self.lst_input[index]))
        input_next = np.load(os.path.join(self.dir_data, self.lst_input[index_next]))

        label = label[:,:,np.newaxis]
        input_prev = input_prev[:,:,np.newaxis]
        input = input[:,:,np.newaxis]
        input_next = input_next[:,:,np.newaxis]
     

        data = {'input': input, 'label': label, 'input_prev': input_prev, 'input_next': input_next}

        if self.transform:
            data = self.transform(data)
            
        return data

class ToTensor(object):
    def __call__(self, data):
        label, input, input_prev, input_next = data['label'], data['input'], data['input_prev'], data['input_next']

        label = label.transpose((2, 0, 1)).astype(np.int64)
        input = input.transpose((2, 0, 1)).astype(np.float32)
        input_prev = input_prev.transpose((2, 0, 1)).astype(np.float32)
        input_next = input_next.transpose((2, 0, 1)).astype(np.float32)
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input), 'input_prev': torch.from_numpy(input_prev), 'input_next': torch.from_numpy(input_next)}

        return data

class Normalization(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input, input_prev, input_next = data['label'], data['input'], data['input_prev'], data['input_next']

        input = (input - self.mean) / self.std
        input_prev = (input_prev - self.mean) / self.std
        input_next = (input_next - self.mean) / self.std

        data = {'label': label, 'input': input, 'input_prev': input_prev, 'input_next': input_next}

        return data