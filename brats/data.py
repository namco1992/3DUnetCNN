import os
import sys
import glob

import tables
import numpy as np
import pandas as pd

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(HOME)

from config import config
from utils import find_downsized_info, read_image_files
from normalize import normalize_data_storage


def create_binary_category_dataset_csv():
    raw_data = {'name': [], 'category': []}
    assert len(config['categories']) == 2
    for i, category in enumerate(config['categories']):
        for subject_dir in glob.glob(os.path.join(config['data_dir'], category, '*')):
            raw_data['name'].append(os.path.basename(subject_dir))
            raw_data['category'].append(i)
    df = pd.DataFrame(raw_data, columns=['name', 'category'])
    df.to_csv(config['csv_file'], header=False, index=False)


def fetch_training_data_files():
    training_data_files = list()
    for subject_dir in glob.glob(os.path.join(config["data_dir"], "*", "*")):
        subject_files = list()
        for modality in config["training_modalities"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    return training_data_files


def create_data_file(out_file, nb_channels, nb_samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, nb_channels] + list(image_shape))
    # truth_shape = tuple([0, 1] + list(image_shape))
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=nb_samples)
    print("data_storage", data_storage)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=((0,)),
                                            filters=filters, expectedrows=nb_samples)
    return hdf5_file, data_storage, truth_storage


def write_image_data_to_file(image_files, data_storage, truth_storage, image_shape, n_channels, crop=None,
                             truth_dtype=np.uint8):
    for set_of_files in image_files:
        subject_data = read_image_files(set_of_files, image_shape, crop=crop)
        data_storage.append(subject_data[np.newaxis])
        truth_storage.append(np.asarray([1] if 'HGG' in set_of_files[0] else [0]))
        # truth_storage.append(np.asarray(subject_data[n_channels][np.newaxis][np.newaxis], dtype=truth_dtype))
    return data_storage, truth_storage


def write_data_to_file(training_data_files, out_file, image_shape, truth_dtype=np.uint8):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param training_data_files: List of tuples containing the training data files. The modalities should be listed in
    the same order in each tuple. The last item in each tuple must be the labeled image.
    Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'),
              ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    :param out_file: Where the hdf5 file will be written to.
    :param image_shape: Shape of the images that will be saved to the hdf5 file.
    :param truth_dtype: Default is 8-bit unsigned integer.
    :return: Location of the hdf5 file with the image data written to it.
    """
    n_samples = len(training_data_files)
    n_channels = len(training_data_files[0])

    try:
        hdf5_file, data_storage, truth_storage = create_data_file(out_file, nb_channels=n_channels,
                                                                  nb_samples=n_samples, image_shape=image_shape)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e

    crop_slices, affine, header = find_downsized_info(training_data_files, image_shape)
    hdf5_file.create_array(hdf5_file.root, "affine", affine)
    write_image_data_to_file(training_data_files, data_storage, truth_storage, image_shape, crop=crop_slices,
                             truth_dtype=truth_dtype, n_channels=n_channels)
    normalize_data_storage(data_storage)
    hdf5_file.close()
    return out_file


if __name__ == '__main__':
    training_data_files = fetch_training_data_files()
    print(len(training_data_files))
    write_data_to_file(training_data_files, config["hdf5_file"], image_shape=config["image_shape"])
