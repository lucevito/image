import os
import uuid
import glob
from PIL import Image
from matplotlib import pyplot as plt

train_images_path = 'Immagini_satellitari/Train/images'
train_masks_path = 'Immagini_satellitari/Train/masks'
test_images_path = 'Immagini_satellitari/Test/images'
test_masks_path = 'Immagini_satellitari/Test/masks'

encoder_filters = [32, 64, 128, 256]
decoder_filters = encoder_filters[::-1]
kernel = 3
num_classes = 2

new_size = (128, 128)
canali_selezionati = [0,1,2,3,4,5,6,7,8,9]
input_shape = (new_size[0], new_size[1], len(canali_selezionati))


train_images_files = glob.glob(train_images_path + '/*.npy')
train_masks_files = glob.glob(train_masks_path + '/*.npy')

def visualize_pixel_plots(image, mask, pred, output_folder='output/'):
    os.makedirs(output_folder, exist_ok=True)
    for i in range(len(image)):
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


def visualize(image, mask, output_folder='bigger/'):
    os.makedirs(output_folder, exist_ok=True)
    for i in range(len(image)):
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
def resize_image(image,new_size):
    image = Image.fromarray(image, mode='L')
    resized_image = image.resize(new_size, Image.BILINEAR)
    resized_array = np.array(resized_image)
    return resized_array
