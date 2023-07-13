
from sklearn.model_selection import train_test_split
from keras import metrics
import matplotlib.pyplot as plt
from keras.utils import plot_model

train_images_path = 'Immagini_satellitari/Train/images'
train_masks_path = 'Immagini_satellitari/Train/masks'
train_images_files = glob.glob(train_images_path + '/*.npy')
train_masks_files = glob.glob(train_masks_path + '/*.npy')
import numpy as np
def add_sample_weights(label,weights):
    # The weights for each class, with the constraint that:
    #     sum(class_weights) == 1.0
    class_weights = tf.constant(weights)
    class_weights = class_weights / tf.reduce_sum(class_weights)
    # Create an image of `sample_weights` by using the label at each pixel as an 
    # index into the `class weights` .
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
    return sample_weights

model = create_unet(input_shape, num_classes, kernel,pool_size, encoder_filters, decoder_filters)
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


epoch = 300
batch_size = 20

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
train_masks_exp = tf.expand_dims(train_masks, axis=-1)
model.fit(train_images, train_masks_exp, epochs=epoch, batch_size=batch_size,
          callbacks=[early_stopping],sample_weight=add_sample_weights(train_masks_exp,weights),
          validation_data=(val_images, val_masks))


model.save(modello)
