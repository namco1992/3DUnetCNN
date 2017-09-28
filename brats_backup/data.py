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


def fetch_training_data_files(category='*'):
    training_data_files = list()
    training_data_names = list()
    for subject_dir in glob.glob(os.path.join(config['data_dir'], category, '*')):
        subject_files = list()
        training_data_names.append(os.path.basename(subject_dir))
        for modality in config['training_modalities']:
            subject_files.append(os.path.join(subject_dir, modality + '.nii.gz'))
        training_data_files.append(tuple(subject_files))
    return training_data_files, training_data_names


def create_data_file(image_set, out_file, crop, nb_channels, image_shape, affine):
    with tables.open_file(out_file, mode='w') as hdf5_file:
        subject_data = read_image_files(image_set, image_shape, crop=crop)
        data = hdf5_file.create_array(
            hdf5_file.root, 'data', subject_data[:nb_channels][np.newaxis], atom=tables.Float32Atom())
        hdf5_file.create_array(hdf5_file.root, 'affine', affine)
        print(data)
    return


# def write_image_data_to_file(image_files, data_storage, image_shape, n_channels, crop=None):
#     for set_of_files in image_files:
#         subject_data = read_image_files(set_of_files, image_shape, crop=crop)
#         data_storage.append(subject_data[:n_channels][np.newaxis])
#     return data_storage


def write_data_to_file():
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
    create_binary_category_dataset_csv()
    training_data_files, training_data_names = fetch_training_data_files()
    training_data_files = training_data_files[:3]
    n_channels = len(training_data_files[0]) - 1
    image_shape = config['image_shape']
    data_dir = config['data_dir']
    crop_slices, affine, header = find_downsized_info(training_data_files, image_shape)

    for i, image_set in enumerate(training_data_files):
        out_file = data_dir + "/hdf5/{}.hdf5".format(training_data_names[i])
        try:
            create_data_file(
                image_set, out_file, crop=crop_slices, nb_channels=n_channels, image_shape=image_shape, affine=affine)
        except Exception as e:
            # If something goes wrong, delete the incomplete data file
            print("Error occured when generate {}".format(out_file))
            # os.remove(out_file)
            raise e

    normalize_data_storage()


if __name__ == '__main__':
    write_data_to_file()
