# Import necessary modules and functions
from dataset import load_dataset, create_smote_dataset, create_sample_dataset
from dataset import unet_load, transform

from train import gr_search, sample_grindseach_learn
from train import unet_train, unet_train_focal
from unet_augmentation import augmentation

from test import test, sample_test
from test import unet_predict_save

from utility import create_output_directory, create_subdirectories


# Parameters for GridSearch
param_grid = {
    "class_weight": [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}, "balanced"],
    "max_depth": [7, 8, 9, 10],
    "max_samples": [0.8, 0.9, 1.0],
    'criterion': ['entropy', 'gini', 'log_loss'],
    "max_features": ["sqrt", "log2"]
}

# Select channels for image data
selected_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Train path
train_path = 'Immagini_satellitari/Train/'
# Test path
test_path = 'Immagini_satellitari/Test/'

# Create the main output directory
path = 'output'
create_output_directory(path)

# Create subdirectories 'sample', 'smote' and 'unet'
subdirectories = ['sample', 'smote', 'unet']
create_subdirectories(path, subdirectories)

# Load training and testing datasets
print("Loading train...")
trainX, trainY = load_dataset(train_path)
print("Loading test...")
testX, testY = load_dataset(test_path)

# Define model names and paths for Random Forest Grid Search
model_name = "rf_GridSearch_model.h"
path = 'output/'
# Perform Grid Search for Random Forest
gr_search(param_grid, trainX, trainY, model_name, path)
# Test the Random Forest model
test(trainX, trainY, testX, testY, model_name, path)

# Define model names and paths for Sample-based models
model_name_sample = "sample_model_"
path = 'output/sample/'
# Create sample datasets and perform Grid Search for each sample-based model
create_sample_dataset(trainX, trainY, model_name_sample, path, n_values=range(1, 11))
sample_grindseach_learn(param_grid, model_name_sample, path, start=1)
# Test sample-based models
sample_test(model_name_sample, trainX, trainY, testX, testY, path)

# Define model names and paths for SMOTE-based models
model_name_smote = "smote_model_"
path = 'output/smote/'
# Create SMOTE datasets and perform Grid Search for each SMOTE-based model
create_smote_dataset(trainX, trainY, model_name_smote, path)
sample_grindseach_learn(param_grid, model_name_smote, path, start=1)
# Test SMOTE-based models
sample_test(model_name_smote, trainX, trainY, testX, testY, path)

# Load and preprocess data for U-Net models
print("Loading train...")
train_images, train_masks = unet_load(train_path, selected_channels)
print("Loading test...")
test_images, test_masks = unet_load(test_path, selected_channels)

# Apply data augmentation and data transformation
augmented_images, augmented_masks = augmentation(train_images, train_masks)
augmented_images, augmented_masks = transform(augmented_images, augmented_masks)
train_images, train_masks = transform(train_images, train_masks)
test_images, test_masks = transform(test_images, test_masks)

# Define path for U-Net models
path = 'output/unet/'

# Train and evaluate the U-Net model without augmentation
model_name = "model_unet"
param = "no_augmentation"
unet_train(model_name, train_images, train_masks, path, len(selected_channels))
unet_predict_save(model_name, "test", param, test_images, test_masks, path)
unet_predict_save(model_name, "train", param, train_images, train_masks, path)

# Train and evaluate the U-Net model with augmentation
model_name = "model_unet_augmentation"
param = "augmentation"
unet_train(model_name, augmented_images, augmented_masks, path, len(selected_channels))
unet_predict_save(model_name, "test", param, test_images, test_masks, path)
unet_predict_save(model_name, "train", param, train_images, train_masks, path)

# Train and evaluate the U-Net model with focal loss and varying alpha values
model_name = "model_unet_augmentation_"
param = "augmentation_focalcrossentropy_"
alphas_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
for alpha in alphas_list:
    unet_train_focal(model_name + str(alpha), alpha, augmented_images, augmented_masks, path, len(selected_channels))
    unet_predict_save(model_name + str(alpha), "test", param + str(alpha), test_images, test_masks, path)
    unet_predict_save(model_name + str(alpha), "train", param + str(alpha), train_images, train_masks, path)
