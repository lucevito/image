import model.config as config
import model.model as model

from torch.utils.data import Dataset
import numpy as np

class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# store the image and mask filepaths, and augmentation transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms

	# return the number of total samples contained in the dataset
	def __len__(self):
		return len(self.imagePaths)
	
	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]
		#print(imagePath)
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = np.load(imagePath)
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#print(self.maskPaths[idx])
		mask = np.load(self.maskPaths[idx], 0)
	
		# check to see if we are applying any transformations
		if self.transforms is not None:

			# apply the transformations to both image and its mask
			image = self.transforms(image)
			mask = self.transforms(mask)
		# return a tuple of the image and its mask
		return (image, mask)
	
