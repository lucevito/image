import glob
import numpy as np

# Load the datasets
def load_dataset(directory):
    images_files = glob.glob(directory + '/images' + '/*.npy')
    masks_files = glob.glob(directory + '/masks' + '/*.npy')
    x = np.array([np.load(file) for file in images_files])
    y = np.array([np.load(file) for file in masks_files])
    x = x.reshape(len(x) * len(x[0]) * len(x[0][0]), 10)
    y = y.reshape(len(y) * len(y[0]) * len(y[0][0]), 1)
    y = np.ravel(y)
    return x, y

# Select all pixels of the target class given as input
# Returns the dataset containing only a specific class
def select_set(x, y, target_class):
    mask = (y == target_class)
    selection_x = x[mask]
    selection_y = y[mask]
    return selection_x, selection_y

# Perform deterministic sampling from the dataset
def deterministic_sampling(x, y, n, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(len(x), n, replace=False)
    sample_x = x[indices]
    sample_y = y[indices]
    return sample_x, sample_y

# Concatenate two datasets along a specified axis
def concatenate(x1, x2, y1, y2):
    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    return x, y

# Reduce the dataset for sampling purposes
def reduce_for_sample(train_x, train_y, n):
    x1, y1 = select_set(train_x, train_y, 1)
    x0, y0 = select_set(train_x, train_y, 0)
    x0, y0 = deterministic_sampling(x0, y0, len(x1) * n)
    x, y = concatenate(x1, x0, y1, y0)
    return x, y

