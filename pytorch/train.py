
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
from torchsummary import summary
import matplotlib.pyplot as plt
import torch
import time
import os

unet = UNET().to(DEVICE)
imagePaths = sorted(list(paths.list_files(train_images_path)))
maskPaths = sorted(list(paths.list_files(train_masks_path)))

split = train_test_split(imagePaths, maskPaths,
    test_size=TEST_SPLIT, random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]
transforms = transforms.Compose([transforms.ToTensor()])

trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
    transforms=transforms)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
    batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
    num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,
    batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
    num_workers=os.cpu_count())


# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=INIT_LR)
# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // BATCH_SIZE
testSteps = len(testDS) // BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}


# Inizializza variabili per l'early stopping
best_test_loss = float('inf')
patience = 10
counter = 0


# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(NUM_EPOCHS)):
    # set the model in training mode
    unet.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    correctTrain = 0
    totalTrain = 0
    totalTestLoss = 0
    correctTest = 0
    totalTest = 0
    # loop over the training set
    for (i, (x, y)) in enumerate(trainLoader):
        # send the input to the device
        (x, y) = (x.to(DEVICE), y.to(DEVICE))
        # perform a forward pass and calculate the training loss
        pred = unet(x)
        loss = lossFunc(pred, y)
        # calculate the accuracy
        predicted_labels = torch.sigmoid(pred) > THRESHOLD
        correctTrain += (predicted_labels == y).sum().item()
        totalTrain += y.numel()
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far
        totalTrainLoss += loss.item()
    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        unet.eval()
        # loop over the validation set
        for (x, y) in testLoader:
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            # make the predictions and calculate the validation loss
            pred = unet(x)
            totalTestLoss += lossFunc(pred, y).item()
            # calculate the accuracy
            predicted_labels = torch.sigmoid(pred) > THRESHOLD
            correctTest += (predicted_labels == y).sum().item()
            totalTest += y.numel()
    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgTestLoss = totalTestLoss / testSteps
    # calculate the accuracy
    avgTrainAcc = (correctTrain / totalTrain) * 100
    avgTestAcc = (correctTest / totalTest) * 100
    # update our training history
    H["train_loss"].append(avgTrainLoss)
    H["test_loss"].append(avgTestLoss)
    H["train_acc"].append(avgTrainAcc)
    H["test_acc"].append(avgTestAcc)
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
    print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))
    print("Train accuracy: {:.2f}%, Test accuracy: {:.2f}%".format(avgTrainAcc, avgTestAcc))
    # Early stopping
    if avgTestLoss < best_test_loss:
        best_test_loss = avgTestLoss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["test_acc"], label="test_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(PLOT_PATH)
# serialize the model to disk
torch.save(unet, MODEL_PATH)
