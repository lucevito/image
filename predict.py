
from model import config
import torch

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from model.model import UNet


def predict_pixels(model, images_path, masks_path):
    torch.save(model.state_dict(), "modello.pt")

    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))

    X_test = []
    y_test = []
    y_pred = []

    for image_file, mask_file in zip(image_files, mask_files):
        image_path = os.path.join(images_path, image_file)
        mask_path = os.path.join(masks_path, mask_file)

        image = plt.np.load(image_path)
        mask = plt.np.load(mask_path)

        X_test.append(image)
        y_test.append(mask)

        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)
        image = image.float()
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(image)
        predMask = torch.sigmoid(predMask)
        predMask = (predMask > config.THRESHOLD) * 32
        y_pred.append(predMask)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_pred = np.stack(y_pred, axis=0)

    # Calcolo della matrice di confusione
    confusion_mat = confusion_matrix(y_test.flatten(), y_pred.flatten())
    print(confusion_mat)

    # Plot dei pixel predetti per ogni immagine
    fig, axes = plt.subplots(nrows=len(image_files), ncols=2, figsize=(10, 2*len(image_files)))
    for i in range(1): #(len(image_files)):
        image = X_test[i]
        true_mask = y_test[i]
        pred_mask = y_pred[i]

        axes[i, 0].imshow(image[..., 0:1])  # Visualizza solo i primi 3 canali dell'immagine
        axes[i, 0].axis('off')
        axes[i, 0].set_title('True Mask')

        axes[i, 1].imshow(pred_mask.squeeze()[..., 0:1])  # Visualizza solo i primi 3 canali della maschera predetta
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Predicted Mask')

    plt.tight_layout()
    plt.show()

    return confusion_mat

model_file = "modello.pt"  # Percorso del file del modello

# Verifica se il file esiste
if os.path.exists(model_file):
    # Crea un'istanza del modello UNet
    unet = UNet()

    # Carica i pesi del modello dal file
    unet.load_state_dict(torch.load(model_file))
    unet.eval()  # Imposta il modello in modalità di valutazione
else:
    print("Il file del modello non esiste.")
    from train import unet


predict_pixels(unet, config.IMAGE_TEST_DATASET_PATH, config.MASK_TEST_DATASET_PATH)
