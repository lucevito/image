import os
import numpy as np
import tensorflow as tf
from model import *
from config import *

if os.path.exists('modello7.h5'):
    model = tf.keras.models.load_model('modello7.h5')
else:
    encoder_filters = [16, 32, 64, 128]
    decoder_filters = encoder_filters[::-1]
    kernel = 3
    input_shape = (256, 256, 10)
    num_classes = 2

    model = create_unet(input_shape, num_classes, kernel, encoder_filters, decoder_filters)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # metrics_callback = MetricsCallback(train_images_resized, train_masks_resized, test_images_resized, test_masks_resized)
    # training_callback = TrainingCallback(validation_data=(train_images_resized, train_masks_resized))
    # , callbacks=[metrics_callback]
    model.fit(train_images_resized, train_masks_resized, epochs=2, batch_size=5)
    model.save('modello7.h5')

test_loss, test_accuracy = model.evaluate(test_images_resized, test_masks_resized)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

predicted_masks = model.predict(test_images_resized)
new_size = (32, 32)
predicted_masks = tf.image.resize(predicted_masks, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
predicted_masks = np.argmax(predicted_masks, axis=-1)

true_masks = test_masks.flatten()
pred_masks = predicted_masks.flatten()

print("============================================================================")
print("Prediction : \n")
print("Evaluation Metrics:")
print("Precision:", precision_score(true_masks, pred_masks, zero_division=0))
print("Recall:", recall_score(true_masks, pred_masks))
print("F1-score:", f1_score(true_masks, pred_masks))
print("Mean IoU:", jaccard_score(true_masks, pred_masks, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(true_masks, pred_masks))

output_folder = 'output/'
os.makedirs(output_folder, exist_ok=True)
visualize_pixel_plots(test_images, test_masks, predicted_masks, output_folder)
