import os
import sys

import tables
import numpy as np
import pandas as pd
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


def calc_mean_std(data_list, data_dir):
    means = list()
    stds = list()
    for name in data_list:
        with tables.open_file(data_dir + '/hdf5/{}.hdf5'.format(name), mode='w') as hdf5_file:
            data = hdf5_file.get_node(hdf5_file.root, "data")
            means.append(data.mean(axis=(1, 2, 3)))
            stds.append(data.std(axis=(1, 2, 3)))
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    return mean, std


def normalize_data_storage():
    # means = list()
    # stds = list()
    # for index in range(data_storage.shape[0]):
    #     data = data_storage[index]
    #     means.append(data.mean(axis=(1, 2, 3)))
    #     stds.append(data.std(axis=(1, 2, 3)))
    # mean = np.asarray(means).mean(axis=0)
    # std = np.asarray(stds).mean(axis=0)
    df = pd.read_csv(config['csv_file'], header=None)
    data_list = df.values.reshape(df.size,).tolist()
    data_dir = config['data_dir']
    mean, std = calc_mean_std(data_list, config['data_dir'])
    for name in data_list:
        with tables.open_file(data_dir + '/hdf5/{}.hdf5'.format(name), mode='w') as hdf5_file:
            data = hdf5_file.get_node(hdf5_file.root, "data")
            data = normalize_data(data, mean, std)
