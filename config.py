import glob
import os
import uuid
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf

train_images_path = 'Immagini_satellitari/Train/images'
train_masks_path = 'Immagini_satellitari/Train/masks'
test_images_path = 'Immagini_satellitari/Test/images'
test_masks_path = 'Immagini_satellitari/Test/masks'

train_images = np.array([np.load(file) for file in glob.glob(train_images_path + '/*.npy')])
train_masks = np.array([np.load(file) for file in glob.glob(train_masks_path + '/*.npy')])
test_images = np.array([np.load(file) for file in glob.glob(test_images_path + '/*.npy')])
test_masks = np.array([np.load(file) for file in glob.glob(test_masks_path + '/*.npy')])

new_size = (256, 256)
train_images_resized = tf.image.resize(train_images, new_size)
train_masks_resized = np.zeros((train_masks.shape[0], *new_size))
test_images_resized = tf.image.resize(test_images, new_size)
test_masks_resized = np.zeros((test_masks.shape[0], *new_size))
for i in range(train_masks.shape[0]):
    mask = Image.fromarray(train_masks[i])
    mask_enlarged = mask.resize(new_size)
    train_masks_resized[i] = np.array(mask_enlarged)
for i in range(test_masks.shape[0]):
    mask = Image.fromarray(test_masks[i])
    mask_enlarged = mask.resize(new_size)
    test_masks_resized[i] = np.array(mask_enlarged)


def visualize_pixel_plots(image, mask, pred, output_folder):
    for i in range(len(test_images)):
        fig, axs = plt.subplots(3, 4, figsize=(15, 5))
        axs[0][0].imshow(image[i][:, :, 0])
        axs[0][0].set_title('Image channel 1')
        axs[0][1].imshow(image[i][:, :, 1])
        axs[0][1].set_title('Image channel 2')
        axs[0][2].imshow(image[i][:, :, 2])
        axs[0][2].set_title('Image channel 3')
        axs[0][3].imshow(image[i][:, :, 3])
        axs[0][3].set_title('Image channel 4')
        axs[1][0].imshow(image[i][:, :, 4])
        axs[1][0].set_title('Image channel 5')
        axs[1][1].imshow(image[i][:, :, 5])
        axs[1][1].set_title('Image channel 6')
        axs[1][2].imshow(image[i][:, :, 6])
        axs[1][2].set_title('Image channel 7')
        axs[1][3].imshow(image[i][:, :, 7])
        axs[1][3].set_title('Image channel 8')
        axs[2][0].imshow(image[i][:, :, 8])
        axs[2][0].set_title('Image channel 9')
        axs[2][1].imshow(image[i][:, :, 9])
        axs[2][1].set_title('Image channel 10')

        axs[2][2].imshow(mask[i])
        axs[2][2].set_title('Truth Mask')

        axs[2][3].imshow(pred[i])
        axs[2][3].set_title('Predicted Mask')

        plt.tight_layout()
        filename = str(uuid.uuid4()) + '.png'
        plt.savefig(os.path.join(output_folder, filename))
        plt.close(fig)


def visualize(image, mask, output_folder):
    for i in range(len(test_images)):
        fig, axs = plt.subplots(3, 4, figsize=(15, 5))
        axs[0][0].imshow(image[i][:, :, 0])
        axs[0][0].set_title('Image channel 1')
        axs[0][1].imshow(image[i][:, :, 1])
        axs[0][1].set_title('Image channel 2')
        axs[0][2].imshow(image[i][:, :, 2])
        axs[0][2].set_title('Image channel 3')
        axs[0][3].imshow(image[i][:, :, 3])
        axs[0][3].set_title('Image channel 4')
        axs[1][0].imshow(image[i][:, :, 4])
        axs[1][0].set_title('Image channel 5')
        axs[1][1].imshow(image[i][:, :, 5])
        axs[1][1].set_title('Image channel 6')
        axs[1][2].imshow(image[i][:, :, 6])
        axs[1][2].set_title('Image channel 7')
        axs[1][3].imshow(image[i][:, :, 7])
        axs[1][3].set_title('Image channel 8')
        axs[2][0].imshow(image[i][:, :, 8])
        axs[2][0].set_title('Image channel 9')
        axs[2][1].imshow(image[i][:, :, 9])
        axs[2][1].set_title('Image channel 10')

        axs[2][2].imshow(mask[i])
        axs[2][2].set_title('Truth Mask')

        plt.tight_layout()
        filename = str(uuid.uuid4()) + '.png'
        plt.savefig(os.path.join(output_folder, filename))
        plt.close(fig)


os.makedirs('output/', exist_ok=True)
visualize(train_images_resized, train_masks_resized, 'output/')
