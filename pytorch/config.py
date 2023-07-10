import torch
import torch.nn as nn
import os
train_images_path = os.path.join("Immagini_satellitari", "Train","images")
train_masks_path = os.path.join("Immagini_satellitari", "Train","masks")
test_images_path = os.path.join("Immagini_satellitari", "Test","images")
test_masks_path = os.path.join("Immagini_satellitari", "Test","masks")

TEST_SPLIT = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False


INIT_LR = 0.001
NUM_EPOCHS = 400
BATCH_SIZE = 64
INPUT_IMAGE_WIDTH = 32
INPUT_IMAGE_HEIGHT = 32
THRESHOLD = 0.5
BASE_OUTPUT = "output"

MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_model.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

NUM_CHANNELS = 10
NUM_CLASSES = 1

canali_selezionati = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
class_weights = {0: 1, 1: 10}

