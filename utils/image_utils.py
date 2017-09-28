import os
import sys

import nibabel as nib
import numpy as np
from nilearn.image import reorder_img, resample_img, new_img_like
from nilearn.image.image import check_niimg
from nilearn.image.image import _crop_img_to as crop_img_to

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(HOME)
sys.path.append(HOME)


def crop_img(img, rtol=1e-8, copy=True, return_slices=False):
    """Crops img as much as possible
    Will crop img, removing as many zero entries as possible
    without touching non-zero entries. Will leave one voxel of
    zero padding around the obtained non-zero area in order to
    avoid sampling issues later on.
    Parameters
    ----------
    img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        img to be cropped.
    rtol: float
        relative tolerance (with respect to maximal absolute
        value of the image), under which values are considered
        negligeable and thus croppable.
    copy: boolean
        Specifies whether cropped data is copied or not.
    return_slices: boolean
        If True, the slices that define the cropped image will be returned.
    Returns
    -------
    cropped_img: image
        Cropped version of the input image
    """

    img = check_niimg(img)
    data = img.get_data()
    infinity_norm = max(-data.min(), data.max())
    passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                     data > rtol * infinity_norm)

    if data.ndim == 4:
        passes_threshold = np.any(passes_threshold, axis=-1)
    coords = np.array(np.where(passes_threshold))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, data.shape[:3])

    slices = [slice(s, e) for s, e in zip(start, end)]

    if return_slices:
        return slices

    return crop_img_to(img, slices, copy=copy)


def get_affine(in_file):
    return read_image(in_file).affine


def read_image_files(image_files, image_shape=None, crop=None, use_nearest_for_last_file=False):
    """

    :param image_files:
    :param image_shape:
    :param crop:
    :param use_nearest_for_last_file: If True, will use nearest neighbor interpolation for the last file. This is used
    because the last file may be the labels file. Using linear interpolation here would mess up the labels.
    :return:
    """
    image_list = list()
    for i, image_file in enumerate(image_files):
        if (i + 1) == len(image_files) and use_nearest_for_last_file:
            interpolation = "nearest"
        else:
            interpolation = "continuous"
        image_list.append(read_image(image_file, image_shape=image_shape, crop=crop, interpolation=interpolation))

    return np.stack([image.get_data() for image in image_list])


def read_image(in_file, image_shape=None, interpolation='continuous', crop=None):
    print("Reading: {0}".format(in_file))
    image = fix_shape(nib.load(in_file))
    if crop:
        image = crop_img_to(image, crop, copy=True)
    if image_shape:
        return resize(image, new_shape=image_shape, interpolation=interpolation)
    else:
        return image


def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def resize(image, new_shape, interpolation="continuous"):
    input_shape = np.asarray(image.shape, dtype=np.float16)
    ras_image = reorder_img(image, resample=interpolation)
    output_shape = np.asarray(new_shape)
    new_spacing = input_shape / output_shape
    new_affine = np.copy(ras_image.affine)
    new_affine[:3, :3] = ras_image.affine[:3, :3] * np.diag(new_spacing)
    return resample_img(ras_image, target_affine=new_affine, target_shape=output_shape, interpolation=interpolation)


def find_downsized_info(training_data_files, input_shape):
    foreground = get_complete_foreground(training_data_files)
    crop_slices = crop_img(foreground, return_slices=True, copy=True)
    cropped = crop_img_to(foreground, crop_slices, copy=True)
    final_image = resize(cropped, new_shape=input_shape, interpolation="nearest")
    return crop_slices, final_image.affine, final_image.header


def get_complete_foreground(training_data_files):
    for i, set_of_files in enumerate(training_data_files):
        subject_foreground = get_foreground_from_set_of_files(set_of_files)
        if i == 0:
            foreground = subject_foreground
        else:
            foreground[subject_foreground > 0] = 1

    return new_img_like(read_image(training_data_files[0][-1]), foreground)


def get_foreground_from_set_of_files(set_of_files, background_value=0, tolerance=0.00001):
    for i, image_file in enumerate(set_of_files):
        image = read_image(image_file)
        is_foreground = np.logical_or(image.get_data() < (background_value - tolerance),
                                      image.get_data() > (background_value + tolerance))
        if i == 0:
            foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

        foreground[is_foreground] = 1

    return foreground
