import os
import sys

import numpy as np
from nilearn.image import new_img_like

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(HOME)
sys.path.append(HOME)

from config import config
from utils import crop_img, crop_img_to, read_image


def normalize_data(data, mean, std):
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data


def normalize_data_storage(data_storage):
    print('Normalizing data...')
    means = list()
    stds = list()
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        means.append(data.mean(axis=(1, 2, 3)))
        stds.append(data.std(axis=(1, 2, 3)))
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    print("mean: ", mean)
    print("std: ", std)
    for index in range(data_storage.shape[0]):
        data_storage[index] = normalize_data(data_storage[index], mean, std)
    print('Normalization done.')
    return data_storage
