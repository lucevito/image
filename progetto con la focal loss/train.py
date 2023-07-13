from config import *
from model import *
from sklearn.model_selection import train_test_split
from keras import *
import matplotlib.pyplot as plt
from keras.utils import plot_model
import numpy as np
import tensorflow as tf

train_images_path = 'Immagini_satellitari/Train/images'
train_masks_path = 'Immagini_satellitari/Train/masks'
train_images_files = glob.glob(train_images_path + '/*.npy')
train_masks_files = glob.glob(train_masks_path + '/*.npy')

model = create_unet(input_shape, num_classes, kernel,pool_size, encoder_filters, decoder_filters)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryFocalCrossentropy(alpha=0.9, gamma=2), metrics=['accuracy'])
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


epoch = 300
batch_size = 30

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

train_images_batch = np.array([np.load(file) for file in train_images_files])
train_masks_batch = np.array([np.load(file) for file in train_masks_files])
train_images_batch = train_images_batch[:, :, :, canali_selezionati]
train_images, val_images, train_masks, val_masks = train_test_split(train_images_batch,
                                                                    train_masks_batch,
                                                                    test_size=0.2, random_state=42)
model.fit(train_images, train_masks, epochs=epoch, batch_size=batch_size,
          callbacks=[early_stopping],
          validation_data=(val_images, val_masks))


model.save(modello)
