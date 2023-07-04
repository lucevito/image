
from model import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from imutils import paths
from sklearn.metrics import confusion_matrix

def calculate_confusion_matrix(model, imagePaths, maskPaths):
    model.eval()
    predictions = []
    targets = []

    for image, mask in zip(imagePaths, maskPaths):
        image = np.load(image)
        mask = np.load(mask)
        
        image = image.astype("float32") / 255.0
        #image = cv2.resize(image, (32, 32))
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)

        with torch.no_grad():
            predMask = model(image).squeeze()
            predMask = torch.sigmoid(predMask)
            predMask = predMask.cpu().numpy()
            predMask = (predMask > config.THRESHOLD).astype(np.uint8)

        predMask = predMask.flatten()
        mask = mask.flatten()

        predictions.extend(predMask.tolist())
        targets.extend(mask.tolist())

    cm = confusion_matrix(targets, predictions)
    return cm


print("[INFO] load up model...")
print("==========================")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

print("[INFO] load train...")
imagePaths = sorted(list(paths.list_files(config.train_images_path)))
maskPaths = sorted(list(paths.list_files(config.train_masks_path)))     

print("[INFO] calculate confusion matrix...")
cm = calculate_confusion_matrix(unet, imagePaths, maskPaths)
print(cm)

print("[INFO] load test...")
imagePaths = sorted(list(paths.list_files(config.test_images_path)))
maskPaths = sorted(list(paths.list_files(config.test_masks_path)))     

print("[INFO] calculate confusion matrix...")
cm = calculate_confusion_matrix(unet, imagePaths, maskPaths)
print(cm)
