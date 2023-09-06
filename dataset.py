import glob
import os
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
import tensorflow as tf


def load_dataset(directory):
    """
    Load dataset from directory containing images and masks in numpy format.

    Parameters:
        directory (str): The path to the directory containing 'images' and 'masks' subdirectories.

    Returns:
        numpy array, numpy array: Loaded x and y arrays from images and masks, respectively.
    """
    images_files = glob.glob(directory + '/images' + '/*.npy')
    masks_files = glob.glob(directory + '/masks' + '/*.npy')
    x = np.array([np.load(file) for file in images_files])
    y = np.array([np.load(file) for file in masks_files])
    x = x.reshape(len(x) * len(x[0]) * len(x[0][0]), 10)
    y = y.reshape(len(y) * len(y[0]) * len(y[0][0]), 1)
    y = np.ravel(y)
    return x, y


def select_set(x, y, target_class):
    """
    Select data samples of a specific target class.

    Parameters:
        x (numpy array): Input features.
        y (numpy array): Target labels.
        target_class (int): The target class value to select.

    Returns:
        numpy array, numpy array: Selected x and y arrays for the target class.
    """
    mask = (y == target_class)
    selection_x = x[mask]
    selection_y = y[mask]
    return selection_x, selection_y


def deterministic_sampling(x, y, n, seed=42):
    """
    Perform deterministic sampling to obtain a fixed number of samples from x and y.

    Parameters:
        x (numpy array): Input features.
        y (numpy array): Target labels.
        n (int): The number of samples to select.
        seed (int): Seed value for random number generator. Default is 42.

    Returns:
        numpy array, numpy array: Selected x and y arrays after deterministic sampling.
    """
    np.random.seed(seed)
    indices = np.random.choice(len(x), n, replace=False)
    sample_x = x[indices]
    sample_y = y[indices]
    return sample_x, sample_y


def concatenate(x1, x2, y1, y2):
    """
    Concatenate two sets of input features and target labels.

    Parameters:
        x1 (numpy array): First set of input features.
        x2 (numpy array): Second set of input features.
        y1 (numpy array): First set of target labels.
        y2 (numpy array): Second set of target labels.

    Returns:
        numpy array, numpy array: Concatenated x and y arrays.
    """
    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    return x, y


def reduce_for_sample(train_x, train_y, n):
    """
    Reduce the number of samples from the majority class to balance the dataset.

    Parameters:
        train_x (numpy array): Input features for training.
        train_y (numpy array): Target labels for training.
        n (int): The reduction factor to apply to the majority class.

    Returns:
        numpy array, numpy array: Reduced x and y arrays to balance the dataset.
    """
    x1, y1 = select_set(train_x, train_y, 1)
    x0, y0 = select_set(train_x, train_y, 0)
    x0, y0 = deterministic_sampling(x0, y0, len(x1) * n)
    x, y = concatenate(x1, x0, y1, y0)
    return x, y


def create_smote_dataset(x, y, filename, path, n_values=range(2, 11)):
    """
    Create datasets with augmented samples using SMOTE.
    Save in a file the dataset

    Parameters:
        x (numpy array): Input features for training.
        y (numpy array): Target labels for training.
        filename (str): The filename to save the augmented datasets.
        path (str): The path where the datasets will be saved.
        n_values (range): A range of values for the SMOTE sampling factor. Default is range(2, 11).
    """
    datasets = {}
    minority_class_size = np.sum(y == 1)
    majority_class_size = np.sum(y == 0)
    for n in n_values:
        sampling_strategy = (minority_class_size / majority_class_size) * n
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        x_train_aug, y_train_aug = smote.fit_resample(x, y)
        datasets[f"TrainAug_{n}"] = (x_train_aug, y_train_aug)
    joblib.dump(datasets, path + 'Dataset_' + filename)


def create_sample_dataset(x, y, filename, path, n_values=range(1, 11)):
    """
    Create datasets with reduced samples from the majority class.
    Save in a file the dataset

    Parameters:
        x (numpy array): Input features for training.
        y (numpy array): Target labels for training.
        filename (str): The filename to save the reduced datasets.
        path (str): The path where the datasets will be saved.
        n_values (range): A range of values for the reduction factor. Default is range(1, 11).
    """
    datasets = {}
    for n in n_values:
        datasets[f"TrainSam_{n}"] = reduce_for_sample(x, y, n)
    joblib.dump(datasets, path + 'Dataset_' + filename)


def create_samplesmote_dataset(x, y, filename, path, n_values=range(2, 11)):
    """
    Create datasets with augmented and reduced samples using SMOTE.
    Save in a file the dataset

    Parameters:
        x (numpy array): Input features for training.
        y (numpy array): Target labels for training.
        filename (str): The filename to save the datasets.
        path (str): The path where the datasets will be saved.
        n_values (range): A range of values for the SMOTE sampling factor. Default is range(2, 11).
    """
    datasets = {}
    minority_class_size = np.sum(y == 1)
    majority_class_size = np.sum(y == 0)
    for n in n_values:
        sampling_strategy = (minority_class_size / majority_class_size) * n
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        x_train_aug, y_train_aug = smote.fit_resample(x, y)
        datasets[f"TrainAug_{n}"] = reduce_for_sample(x_train_aug, y_train_aug, 1)
    joblib.dump(datasets, path + 'Dataset_' + filename)


def load_model_prefix(directory, prefix):
    """
    Load models with a common filename prefix from a directory.

    Parameters:
        directory (str): The path to the directory containing the models.
        prefix (str): The common filename prefix of the models to load.

    Returns:
        list, list: List of file paths and list of corresponding file names.
    """
    file_prefix = []
    file_name = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            file_prefix.append(os.path.join(directory, filename))
            file_name.append(filename)
    return file_prefix, file_name


def unet_load(path, selected_channels):
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
