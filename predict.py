import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from model import config
from model.model import UNet


def visualize_pixel_plots(image, mask, pred):
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

    # Visualizzazione della maschera di ground truth nel secondo subplot
    axs[2][2].imshow(mask)
    axs[2][2].set_title('Truth Mask')

    # Visualizzazione della maschera predetta nel terzo subplot
    axs[2][3].imshow(pred)
    axs[2][3].set_title('Predicted Mask')

    # Impostazione dello spazio tra i subplots
    plt.tight_layout()
    # Visualizzazione della figura
    plt.show()


def predict_pixels(model, images_path, masks_path):
    torch.save(model.state_dict(), "modello.pt")

    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))

    image_test = []
    mask_test = []
    pred_mask = []
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        for image_file, mask_file in zip(image_files, mask_files):
            image_path = os.path.join(images_path, image_file)
            mask_path = os.path.join(masks_path, mask_file)
            image = np.load(image_path)
            mask = np.load(mask_path)

            image = image.astype("float32") / 255.0
            original = image.copy()
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image).to(config.DEVICE)

            pred = unet(image).squeeze()
            pred = torch.sigmoid(pred)
            pred = pred.cpu().numpy()
            pred = (pred > config.THRESHOLD) * 255
            pred = pred.astype(np.uint8)

            print("mask:")
            for i in range(10):
                print(mask[i])
            print("pred: ")
            for i in range(10):
                print(pred[i])

            image_test.append(original)
            mask_test.append(mask)
            pred_mask.append(pred)

        image_test = np.array(image_test)
        mask_test = np.array(mask_test)
        pred_mask = np.array(pred_mask)

        confusion_mat = confusion_matrix(mask_test.flatten(), pred_mask.flatten())
        print(confusion_mat)
        for i in range(len(image_test)):
            print(i)
            visualize_pixel_plots(image_test[i], mask_test[i], pred_mask[i])
            input("Premi Invio")
        #visualize_pixel_plots(image_test[i], mask_test[i], pred_mask[i])



if os.path.exists("modello.pt"):
    unet = UNet()
    unet.load_state_dict(torch.load("modello.pt"))
    unet.eval()  # Imposta il modello in modalità di valutazione
else:
    print("Il file del modello non esiste.")
    from train import unet

predict_pixels(unet, config.IMAGE_TEST_DATASET_PATH, config.MASK_TEST_DATASET_PATH)
