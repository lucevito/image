import numpy as np
import tensorflow as tf


def flip_left_right(input_image, input_mask):
    """
    Flip input image and mask horizontally (left to right).

    Args:
        input_image (tensorflow.Tensor): Input image tensor.
        input_mask (tensorflow.Tensor): Input mask tensor.

    Returns:
        input_image (tensorflow.Tensor): Flipped input image tensor.
        input_mask (tensorflow.Tensor): Flipped input mask tensor.

    """
    channels = tf.split(input_image, num_or_size_splits=10, axis=-1)
    flipped_channels = [tf.image.flip_left_right(channel) for channel in channels]
    input_image = tf.concat(flipped_channels, axis=-1)
    input_mask = tf.image.flip_left_right(input_mask)
    return input_image, input_mask


def rotate_image(input_image, input_mask, angle=45):
    """
    Rotate input image and mask by a specified angle (default is 45 degrees).

    Args:
        input_image (tensorflow.Tensor): Input image tensor.
        input_mask (tensorflow.Tensor): Input mask tensor.
        angle (int): Rotation angle in degrees (default is 45).

    Returns:
        input_image (tensorflow.Tensor): Rotated input image tensor.
        input_mask (tensorflow.Tensor): Rotated input mask tensor.

    """
    channels = tf.split(input_image, num_or_size_splits=10, axis=-1)
    rotate_channels = [tf.image.rot90(channel, k=angle // 90) for channel in channels]

    input_image = tf.concat(rotate_channels, axis=-1)
    input_mask = tf.image.rot90(input_mask, k=angle // 90)
    return input_image, input_mask


def rotation(angle, image, mask, augmented_images, augmented_masks):
    """
    Apply rotation augmentation to the image and mask and append to augmented lists.

    Args:
        angle (int): Rotation angle in degrees.
        image (tensorflow.Tensor): Input image tensor.
        mask (tensorflow.Tensor): Input mask tensor.
        augmented_images (list): List to store augmented images.
        augmented_masks (list): List to store augmented masks.

    Returns:
        augmented_images (list): Updated list of augmented images.
        augmented_masks (list): Updated list of augmented masks.

    """
    augmented_image, augmented_mask = rotate_image(image, mask, angle)
    augmented_images.append(augmented_image)
    augmented_masks.append(augmented_mask)
    return augmented_images, augmented_masks


def augmentation(train_images_batch, train_masks_batch):
    """
    Apply data augmentation to a batch of training images and masks.

    Args:
        train_images_batch (list): List of input training image tensors.
        train_masks_batch (list): List of input training mask tensors.

    Returns:
        augmented_images (list): List of augmented image tensors.
        augmented_masks (list): List of augmented mask tensors.

    """
    augmented_images = []
    augmented_masks = []
    angles = [15, 30, 45, 90, 120, 180, 270]
    for image, mask in zip(train_images_batch, train_masks_batch):
        augmented_images.append(image)
        augmented_masks.append(mask)
        if not np.all(mask == 0):
            augmented_image, augmented_mask = flip_left_right(image, mask)
            augmented_images.append(augmented_image)
            augmented_masks.append(augmented_mask)
            for angle in angles:
                augmented_images, augmented_masks = rotation(angle, image, mask, augmented_images, augmented_masks)
    return augmented_images, augmented_masks
