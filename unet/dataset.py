import glob
import tensorflow as tf

import numpy as np


def load(path, selected_channels):
    """
    Load data from NumPy format images and masks.

    Args:
        path (str): The main path containing "images" and "masks" folders.
        selected_channels (list): List of channel indices to select from images.

    Returns:
        images (numpy.ndarray): Loaded images array with selected channels.
        masks (tensorflow.Tensor): Loaded masks' tensor.

    """
    images_path = path + '/images'
    masks_path = path + '/masks'
    images_files = glob.glob(images_path + '/*.npy')
    masks_files = glob.glob(masks_path + '/*.npy')
    images = np.array([np.load(file) for file in images_files])
    masks = np.array([np.load(file) for file in masks_files])
    images = images[:, :, :, selected_channels]
    masks = tf.expand_dims(masks, axis=-1)
    return images, masks


def normalize(image):
    """
    Normalize an image to the range [0, 1].

    Args:
        image (tensorflow.Tensor): Image tensor to normalize.

    Returns:
        image (tensorflow.Tensor): Normalized image tensor.

    """
    image = tf.cast(image, tf.float32) / 255.0
    return image


def transform(images, masks):
    """
    Apply normalization to images.

    Args:
        images (numpy.ndarray): Array of images to normalize.
        masks (numpy.ndarray): Array of masks.

    Returns:
        transformed_images (numpy.ndarray): Array of normalized images.
        transformed_masks (numpy.ndarray): Array of mask.

    """
    transformed_images = []
    for image in images:
        transformed_images.append(normalize(image))
    transformed_images = np.array(transformed_images)
    transformed_masks = np.array(masks)
    return transformed_images, transformed_masks
