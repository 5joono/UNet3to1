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

        #lst_label.sort()
        #lst_input.sort()

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
        input1 = np.load(os.path.join(self.dir_data, self.lst_input[index_prev]))
        input2 = np.load(os.path.join(self.dir_data, self.lst_input[index]))
        input3 = np.load(os.path.join(self.dir_data, self.lst_input[index_next]))


        if self.transform:
            label = self.transform(label)
            input1 = self.transform(input1)
            input2 = self.transform(input2)
            input3 = self.transform(input3)
        input = torch.cat((input1, input2, input3), axis=-1)

        data = {'input': input, 'label': label}

        return data
