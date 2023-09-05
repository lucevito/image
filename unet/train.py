from unet import build_unet_model
from sklearn.model_selection import train_test_split
import tensorflow as tf


def train_focal(model_name, param, images, masks, path,
                num_channel=10, epoch=500, batch_size=20, patience=200):
    """
    Train a U-Net model with focal loss for image segmentation.

    Args:
        model_name (str): Name of the saved model file.
        param (float): Focal loss parameter.
        images (numpy.ndarray): Input images for training.
        masks (numpy.ndarray): Ground truth masks for training.
        path (str): Directory path where the trained model will be saved.
        num_channel (int): Number of input channels (default is 10).
        epoch (int): Number of training epochs (default is 500).
        batch_size (int): Batch size for training (default is 20).
        patience (int): Number of epochs with no improvement to trigger early stopping (default is 200).

    Returns:
        None

    """
    unet_model = build_unet_model(num_channel)
    unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss=tf.keras.losses.BinaryFocalCrossentropy(
                           apply_class_balancing=True, alpha=param, gamma=2.0),
                       metrics="accuracy")
    epoch = epoch
    batch_size = batch_size
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    train_images, val_images, train_masks, val_masks = train_test_split(images,
                                                                        masks,
                                                                        test_size=0.2, random_state=42)
    unet_model.fit(train_images, train_masks, epochs=epoch, batch_size=batch_size,
                   callbacks=[early_stopping],
                   validation_data=(val_images, val_masks))

    unet_model.save(path + model_name)


def train(model_name, images, masks, path,
          num_channel=10, epoch=500, batch_size=20, patience=200):
    """
    Train a U-Net model for image segmentation using binary cross-entropy loss.

    Args:
        model_name (str): Name of the saved model file.
        images (numpy.ndarray): Input images for training.
        masks (numpy.ndarray): Ground truth masks for training.
        path (str): Directory path where the trained model will be saved.
        num_channel (int): Number of input channels (default is 10).
        epoch (int): Number of training epochs (default is 500).
        batch_size (int): Batch size for training (default is 20).
        patience (int): Number of epochs with no improvement to trigger early stopping (default is 200).

    Returns:
        None

    """
    unet_model = build_unet_model(num_channel)
    unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss='binary_crossentropy',
                       metrics="accuracy")
    epoch = epoch
    batch_size = batch_size
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    train_images, val_images, train_masks, val_masks = train_test_split(images,
                                                                        masks,
                                                                        test_size=0.2, random_state=42)
    unet_model.fit(train_images, train_masks, epochs=epoch, batch_size=batch_size,
                   callbacks=[early_stopping],
                   validation_data=(val_images, val_masks))

    unet_model.save(path + model_name)
