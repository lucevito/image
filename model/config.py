

import torch
import os

# base path of the dataset
DATASET_PATH_TRAIN = os.path.join("Immagini_satellitari", "Train")
IMAGE_TRAIN_DATASET_PATH = os.path.join(DATASET_PATH_TRAIN, "images")
MASK_TRAIN_DATASET_PATH = os.path.join(DATASET_PATH_TRAIN, "masks")

# base path of the dataset
DATASET_PATH_TEST = os.path.join("Immagini_satellitari", "Test")
IMAGE_TEST_DATASET_PATH = os.path.join(DATASET_PATH_TEST, "images")
MASK_TEST_DATASET_PATH = os.path.join(DATASET_PATH_TEST, "masks")

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
#NUM_CHANNELS = 1
#NUM_CLASSES = 1
#NUM_LEVELS = 3

# initialize learning rate
#Learning rate controls how quickly or slowly a neural network model learns a problem
INIT_LR = 0.001

#One epoch leads to underfitting of the curve in the graph
NUM_EPOCHS = 50

#The batch size defines the number of samples that will be propagated through the network.
BATCH_SIZE = 10

# define the input image dimensions
INPUT_IMAGE_WIDTH = 32
INPUT_IMAGE_HEIGHT = 32

# define threshold to filter weak predictions(soglia)
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

