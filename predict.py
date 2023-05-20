import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix

from model import config
from train import unet


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def predict_pixels(model, images_path, masks_path):
    # Preparazione dei dati di test
    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))

    X_test = []
    y_test = []

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
        predMask = predMask.cpu().numpy()
        # filter out the weak predictions and convert them to integers
        predMask = (predMask > config.THRESHOLD) * 32
        predMask = predMask.astype(np.uint8)

    X_test = np.array(X_test)
    y_test = np.array(y_test)


    # Esecuzione della predizione sui dati di test
    #y_pred = model(X_test)

    # Calcolo della matrice di confusione
    confusion_mat = confusion_matrix(y_test.flatten(), y_pred.flatten())

    # Plot dei pixel predetti per ogni immagine
    fig, axes = plt.subplots(nrows=len(image_files), ncols=2, figsize=(10, 2*len(image_files)))
    for i in range(len(image_files)):
        image = X_test[i]
        true_mask = y_test[i]
        pred_mask = y_pred[i]

        axes[i, 0].imshow(image[..., 0:3])  # Visualizza solo i primi 3 canali dell'immagine
        axes[i, 0].axis('off')
        axes[i, 0].set_title('True Mask')

        axes[i, 1].imshow(pred_mask[..., 0:3])  # Visualizza solo i primi 3 canali della maschera predetta
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Predicted Mask')

    plt.tight_layout()
    plt.show()

    return confusion_mat

predict_pixels(unet,config.IMAGE_TEST_DATASET_PATH,config.MASK_TEST_DATASET_PATH)