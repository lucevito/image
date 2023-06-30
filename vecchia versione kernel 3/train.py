from model import *
from config import *

from sklearn.model_selection import train_test_split
from keras import metrics
import matplotlib.pyplot as plt
from keras.utils import plot_model

train_images_path = 'Immagini_satellitari/Train/images'
train_masks_path = 'Immagini_satellitari/Train/masks'

train_images_files = glob.glob(train_images_path + '/*.npy')
train_masks_files = glob.glob(train_masks_path + '/*.npy')

modello = 'modello.h5'
if os.path.exists(modello):
    with tf.keras.utils.custom_object_scope({'weighted_loss': weighted_loss}):
        model = tf.keras.models.load_model(modello)
else:
    model = create_unet(input_shape, num_classes, kernel,pool_size, encoder_filters, decoder_filters)
    model.compile(optimizer='adam', loss=weighted_loss, metrics=['accuracy'])
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

num_parts = 1
epoch = 300
batch_size = 50
ripetizioni = 1

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=10,
    restore_best_weights=True
)

for k in range(ripetizioni):
    for batch_idx in range(num_parts):
        start_idx = batch_idx * len(train_images_files) // num_parts
        end_idx = (batch_idx + 1) * len(train_images_files) // num_parts

        train_images_batch = np.array([np.load(file) for file in train_images_files[start_idx:end_idx]])
        train_masks_batch = np.array([np.load(file) for file in train_masks_files[start_idx:end_idx]])
        train_images_batch = train_images_batch[:, :, :, canali_selezionati]
        train_images, val_images, train_masks, val_masks = train_test_split(train_images_batch,
                                                                            train_masks_batch,
                                                                            test_size=0.2, random_state=42)
        callback = TrainingCallback(val_images, val_masks)
        model.fit(train_images, train_masks, epochs=epoch,
                  batch_size=batch_size, callbacks=[callback,early_stopping])

        print("Parte : ", batch_idx + 1, " finita su :", num_parts)
        print("==============================================")

    model.save('modello.h5')




