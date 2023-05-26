import model.config as config
import model.model as model
import model.dataset as dataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import torch
import time
import os

# load the image and mask filepaths in a sorted manner
imagePathsTrain = sorted(list(paths.list_files(config.IMAGE_TRAIN_DATASET_PATH)))
maskPathsTrain = sorted(list(paths.list_files(config.MASK_TRAIN_DATASET_PATH)))

# define transformations
transforms = transforms.Compose([
    transforms.ToTensor()])

# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
split = train_test_split(imagePathsTrain, maskPathsTrain,
                         test_size=config.TEST_SPLIT, random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]

# create the train and test datasets
trainDS = dataset.SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
                                      transforms=transforms)

testDS = dataset.SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
                                     transforms=transforms)

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")

# Create a custom sampler to balance the dataset
class MaskBalanceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        mask_indices = [i for i, (_, mask) in enumerate(self.dataset) if torch.all(mask == 0)]
        non_mask_indices = [i for i, (_, mask) in enumerate(self.dataset) if not torch.all(mask == 0)]
        num_samples = min(len(mask_indices), len(non_mask_indices))

        indices = mask_indices[:num_samples] + non_mask_indices[:num_samples]
        indices = torch.randperm(len(indices)).tolist()
        return iter(indices)

    def __len__(self):
        return len(self.dataset)

# create the training and test data loaders with balanced sampling
trainLoader = DataLoader(trainDS, sampler=MaskBalanceSampler(trainDS),
                         batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                         num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,
                        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                        num_workers=os.cpu_count())

# initialize our UNet model
unet = model.UNet().to(config.DEVICE)

# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)

# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
    # set the model in training mode
    unet.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalTestLoss = 0

    # loop over the training set
    for (i, (x, y)) in enumerate(trainLoader):
        # send the input to the device
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
        # perform a forward pass and calculate the training loss
        pred = unet(x)
        loss = lossFunc(pred, y)
        """
        if torch.all(y == 0):
            loss = lossFunc(pred, y) * weight
        else:
            loss = lossFunc(pred, y)

        """
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far
        totalTrainLoss += loss

    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        unet.eval()
        # loop over the validation set
        for (x, y) in testLoader:
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            # make the predictions and calculate the validation loss
            pred = unet(x)
            """
            if torch.all(y == 0):
                totalTestLoss += lossFunc(pred, y) * weight
            else:
                totalTestLoss += lossFunc(pred, y)

            """
            totalTestLoss += lossFunc(pred, y)

    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgTestLoss = totalTestLoss / testSteps

    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
    print("Train loss: {:.6f}, Test loss: {:.4f}".format(
        avgTrainLoss, avgTestLoss))

# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
    endTime - startTime))
