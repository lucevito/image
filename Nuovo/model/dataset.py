import numpy as np
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        image = np.load(imagePath)  # Carica l'immagine da un file .npy
        mask = np.load(self.maskPaths[idx])  # Carica la maschera da un file .npy

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return (image, mask)
