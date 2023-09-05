from augmentation import augmentation
from dataset import load, transform
from test import predict_save
from train import train, train_focal
from utility import create_output_directory

# Create the main output directory
output_path = 'output/'
create_output_directory(output_path)

# Select channels for image data
selected_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Load training and test data
train_path = 'Immagini_satellitari/Train'
test_path = 'Immagini_satellitari/Test'
train_images, train_masks = load(train_path, selected_channels)
test_images, test_masks = load(test_path, selected_channels)

# Apply data augmentation and data transformation
augmented_images, augmented_masks = augmentation(train_images, train_masks)
augmented_images, augmented_masks = transform(augmented_images, augmented_masks)
train_images, train_masks = transform(train_images, train_masks)
test_images, test_masks = transform(test_images, test_masks)

# Train and evaluate the U-Net model without augmentation
model_name = "model_unet"
param = "no_augmentation"
train(model_name, train_images, train_masks, output_path, len(selected_channels))
predict_save(model_name, "test", param, test_images, test_masks, output_path)
predict_save(model_name, "train", param, train_images, train_masks, output_path)

# Train and evaluate the U-Net model with augmentation
model_name = "model_unet_augmentation"
param = "augmentation"
train(model_name, augmented_images, augmented_masks, output_path, len(selected_channels))
predict_save(model_name, "test", param, test_images, test_masks, output_path)
predict_save(model_name, "train", param, train_images, train_masks, output_path)

# Train and evaluate the U-Net model with focal loss and varying alpha values
model_name = "model_unet_augmentation_"
param = "augmentation_focalcrossentropy_"
alphas_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
for alpha in alphas_list:
    train_focal(model_name + str(alpha), alpha, augmented_images, augmented_masks, output_path, len(selected_channels))
    predict_save(model_name + str(alpha), "test", param + str(alpha), test_images, test_masks, output_path)
    predict_save(model_name + str(alpha), "train", param + str(alpha), train_images, train_masks, output_path)
