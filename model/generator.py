import os
from random import shuffle

import numpy as np

from utils import pickle_dump, pickle_load
from .augment import augment_data
from brats.config import config

np.random.seed(config['random_seed'])


def get_training_and_validation_generators(data_file, batch_size, train_index, valid_index, augment=False,
                                           augment_flip=True, augment_distortion_factor=0.25):
    """
    Creates the training and validation generators that can be used when training the model.
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param augment: If True, training data will be distorted on the fly so as to avoid over-fitting.
    :param data_file: hdf5 file to load the data from.
    :param batch_size: Size of the batches that the training generator will provide.
    :param training_keys_file: Pickle file where the index locations of the training data will be stored.
    :param validation_keys_file: Pickle file where the index locations of the validation data will be stored.
    :param data_split: How the training and validation data will be split. 0 means all the data will be used for
    validation and none of it will be used for training. 1 means that all the data will be used for training and none
    will be used for validation. Default is 0.8 or 80%.
    :param overwrite: If set to True, previous files will be overwritten. The default mode is false, so that the
    training and validation splits won't be overwritten when rerunning model training.
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """
    # training_list, validation_list = get_validation_split(data_file, data_split=data_split, overwrite=overwrite,
    #                                                       training_file=training_keys_file,
    #                                                       testing_file=validation_keys_file)
    training_generator = data_generator(data_file, train_index, batch_size=batch_size, augment=augment, augment_flip=augment_flip,
                                        augment_distortion_factor=augment_distortion_factor)
    validation_generator = data_generator(data_file, valid_index, batch_size=batch_size)
    # Set the number of training and testing samples per epoch correctly
    num_training_steps = len(train_index) // batch_size
    num_validation_steps = len(valid_index)
    return training_generator, validation_generator, num_training_steps, num_validation_steps


def get_validation_split(data_file, training_file, testing_file, data_split=0.8, overwrite=False):
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = data_file.root.data.shape[0]
        sample_list = list(range(nb_samples))
        training_list, testing_list = split_list(sample_list, split=data_split)
        pickle_dump(training_list, training_file)
        pickle_dump(testing_list, testing_file)
        return training_list, testing_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(testing_file)


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def data_generator(data_file, index_list, batch_size=1, augment=False, augment_flip=True,
                   augment_distortion_factor=0.25):
    while True:
        x_list = list()
        y_list = list()
        shuffle(index_list)
        for index in index_list:
            add_data(x_list, y_list, data_file, index, augment=augment, augment_flip=augment_flip,
                     augment_distortion_factor=augment_distortion_factor)
            if len(x_list) == batch_size:
                yield np.asarray(x_list), np.asarray(y_list)
                x_list = list()
                y_list = list()


def add_data(x_list, y_list, data_file, index, augment=False, augment_flip=True,
             augment_distortion_factor=0.25):
    """
    Adds data from the data file to the given lists of feature and target data
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
    augment_distortion_factor)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :return:
    """
    data = data_file.root.data[index]
    truth = data_file.root.truth[index]
    if augment:
        data = augment_data(
            data, data_file.root.affine, flip=augment_flip, scale_deviation=augment_distortion_factor)

    # Covert channel first to channel last
    x_list.append(data.T)
    y_list.append([truth])


# def convert_data(x_list, y_list, n_labels=1, labels=None):
#     x = np.asarray(x_list)
#     y = np.asarray(y_list)
#     if n_labels == 1:
#         y[y > 0] = 1
#     elif n_labels > 1:
#         y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
#     return x, y


# def get_multi_class_labels(data, n_labels, labels=None):
#     new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
#     y = np.zeros(new_shape, np.int8)
#     for label_index in range(n_labels):
#         if labels:
#             y[:, label_index][data[:, 0] == labels[label_index]] = 1
#         else:
#             y[:, label_index][data[:, 0] == (label_index + 1)] = 1
#     return y
