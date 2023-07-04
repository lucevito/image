
import torch
import os
train_images_path = os.path.join("Immagini_satellitari", "Train","images")
train_masks_path = os.path.join("Immagini_satellitari", "Train","masks")
test_images_path = os.path.join("Immagini_satellitari", "Test","images")
test_masks_path = os.path.join("Immagini_satellitari", "Test","masks")

# define the test split
TEST_SPLIT = 0.15
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 400
BATCH_SIZE = 64
# define the input image dimensions
INPUT_IMAGE_WIDTH = 32
INPUT_IMAGE_HEIGHT = 32
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_model.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])


# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3
