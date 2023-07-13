import os
import uuid
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def visualize_pixel_plots(image, mask, pred, output_folder):
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)
    fig, axs = plt.subplots(3, 4, figsize=(15, 5))
    axs[0][0].imshow(image[:, :, 0])
    axs[0][0].set_title('Image channel 1')
    axs[0][1].imshow(image[:, :, 1])
    axs[0][1].set_title('Image channel 2')
    axs[0][2].imshow(image[:, :, 2])
    axs[0][2].set_title('Image channel 3')
    axs[0][3].imshow(image[:, :, 3])
    axs[0][3].set_title('Image channel 4')
    axs[1][0].imshow(image[:, :, 4])
    axs[1][0].set_title('Image channel 5')
    axs[1][1].imshow(image[:, :, 5])
    axs[1][1].set_title('Image channel 6')
    axs[1][2].imshow(image[:, :, 6])
    axs[1][2].set_title('Image channel 7')
    axs[1][3].imshow(image[:, :, 7])
    axs[1][3].set_title('Image channel 8')
    axs[2][0].imshow(image[:, :, 8])
    axs[2][0].set_title('Image channel 9')
    axs[2][1].imshow(image[:, :, 9])
    axs[2][1].set_title('Image channel 10')

    axs[2][2].imshow(mask)
    axs[2][2].set_title('Truth Mask')

    axs[2][3].imshow(pred)
    axs[2][3].set_title('Predicted Mask')

    plt.tight_layout()
    filename = str(uuid.uuid4()) + '.png'
    plt.savefig(os.path.join(output_folder, filename))
    plt.close(fig)

encoder_filters = [64, 128, 256, 512]
decoder_filters = encoder_filters[::-1]
kernel = 3
num_classes = 1
pool_size = (2, 2)

size = (32, 32)
canali_selezionati = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
input_shape = (size[0], size[1], len(canali_selezionati))
weights = [0.60, 0.40]
modello = '040.h5'
