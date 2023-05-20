import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from model import config
from train import unet

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os


def prepare_plot(origImage, origMask, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()


# set model to evaluation mode

def make_predictions(model, imagePath):
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = np.load(imagePath)
        # resize the image and make a copy of it for visualization
        orig = image.copy()
        # find the filename and generate the path to ground truth
        # mask
        filename = imagePath.split(os.path.sep)[-1]
        groundTruthPath = os.path.join(config.MASK_TEST_DATASET_PATH,
                                       filename)
        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        gtMask = np.load(groundTruthPath,0)

    # make the channel axis to be the leading one, add a batch
    # dimension, create a PyTorch tensor, and flash it to the
    # current device

    image = torch.from_numpy(image).to(config.DEVICE)
    # make the prediction, pass the results through the sigmoid
    # function, and convert the result to a NumPy array
    predMask = model(image)
    # filter out the weak predictions and convert them to integers
    predMask = (predMask > config.THRESHOLD) * 32
    predMask = predMask.astype(np.uint8)
    # prepare a plot for visualization
    prepare_plot(orig, gtMask, predMask)


# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
# iterate over the randomly selected test image paths
for path in imagePaths:
    # make predictions and visualize the results
    make_predictions(unet, path)
