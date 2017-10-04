import os
import sys
from random import shuffle

import tables
import numpy as np
from keras.models import load_model

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(HOME)
sys.path.append(HOME)

from config import config
from model import build_model
from utils.basic_utils import pickle_dump, pickle_load


def predict(model, x, batch_size=8, verbose=2):
    p = model.predict(x, batch_size=batch_size, verbose=verbose)
    return p


def evaluate(y_true, y_pred):
    print("Truth: {}".format(y_true))
    print("Prediction: {}".format(y_pred))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    result = (y_pred == y_true)
    acc = np.count_nonzero(result) / y_true.size
    print("acc: {}".format(acc))
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))

    print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP, FP, TN, FN))


def add_data(x_list, y_list, data_file, index):
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
    # Covert channel first to channel last
    x_list.append(data.T)
    y_list.append(truth)


def get_random_dataset(data_file, size, output_file, index_file=None):
    x_list = list()
    y_list = list()
    if index_file:
        index_list = pickle_load(index_file)
    else:
        nb_samples = data_file.root.data.shape[0]
        sample_list = list(range(nb_samples))
        shuffle(sample_list)
        index_list = sample_list[:size]
    print("dataset index: {}".format(index_list))
    for index in index_list:
        add_data(x_list, y_list, data_file, index)
    pickle_dump(index_list, output_file)
    return x_list, y_list


def main():
    with tables.open_file(config["hdf5_file"], "r") as hdf5_file_opened:
        x, y_true = get_random_dataset(hdf5_file_opened, 50, './evaluate_index.pkl')
        weight_path = config['model_file'].format(1)
        print('Load weights from {}'.format(weight_path))
        m = build_model(
            input_shape=config["input_shape"], initial_learning_rate=config["initial_learning_rate"], weights=weight_path)
        y_pred = predict(m, x)
        evaluate(y_true, y_pred)


if __name__ == '__main__':
    main()
