import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as ni

dir_input = '/home/joono/Desktop/data_LITS/volume'
dir_label = '/home/joono/Desktop/data_LITS/segmentation'
dir_data = './dataset'
len_input = len(os.listdir(dir_input))
len_label = len(os.listdir(dir_label))

assert len_input == len_label, 'input and label are not matched'

len_train = len_input * 3 // 5
len_val = len_input // 5
len_test = len_input - len_train - len_val

dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)
if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)
if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

for index in np.arange(len_input):
    name_input = f'volume-{index}.nii'
    name_label = f'segmentation-{index}.nii'
    img_input = ni.load(os.path.join(dir_input, name_input)).get_fdata()
    img_label = ni.load(os.path.join(dir_label, name_label)).get_fdata()

    if index in np.arange(len_train):
        dir_save = dir_save_train
    elif index in np.arange(len_train + len_val):
        dir_save = dir_save_val
    else:
        dir_save = dir_save_test
    for slice in np.arange(img_input.shape[-1]):
        np.save(os.path.join(dir_save, f'input_{index:03d}_{slice:04d}.npy'), img_input[:,:,slice])
        np.save(os.path.join(dir_save, f'label_{index:03d}_{slice:04d}.npy'), img_label[:,:,slice])
    print(f'sample {index} saved in {dir_save}')    
