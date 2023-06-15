import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, jaccard_score, confusion_matrix, f1_score
import tensorflow as tf


def weighted_loss(y_true, y_pred):
    weights = tf.where(tf.equal(y_true, 0), 1.0, 1.0)  # falso,vero pesi
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    weighted_loss = tf.reduce_mean(loss * weights)
    return weighted_loss


def create_unet(input_shape, num_classes, kernel, encoder_filters, decoder_filters):
    inputs = tf.keras.Input(input_shape)
    x = inputs

    conv_layers = []
    pooling_layers = []
    for filters in encoder_filters:
        conv = tf.keras.layers.Conv2D(filters, kernel, activation='relu', padding='same')(x)
        conv = tf.keras.layers.Conv2D(filters, kernel, activation='relu', padding='same')(conv)
        pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
        conv_layers.append(conv)
        pooling_layers.append(pool)
        x = pool

    bottleneck = tf.keras.layers.Conv2D(encoder_filters[-1], kernel,
                                        activation='relu', padding='same')(pooling_layers[-1])
    bottleneck = tf.keras.layers.Conv2D(encoder_filters[-1], kernel,
                                        activation='relu', padding='same')(bottleneck)

    for filters in reversed(decoder_filters):
        up = tf.keras.layers.UpSampling2D(size=(2, 2))(bottleneck)
        concat = tf.keras.layers.Concatenate()([up, conv_layers.pop()])
        conv = tf.keras.layers.Conv2D(filters, kernel, activation='relu', padding='same')(concat)
        conv = tf.keras.layers.Conv2D(filters, kernel, activation='relu', padding='same')(conv)
        bottleneck = conv

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid')(bottleneck)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_images, validation_masks):
        super(TrainingCallback, self).__init__()
        self.validation_images = validation_images
        self.validation_masks = validation_masks

    def on_epoch_end(self, epoch, logs=None):
        val_loss, val_acc = self.model.evaluate(self.validation_images, self.validation_masks)
        print(f'Epoch {epoch + 1} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.4f}')


class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_images, train_masks, test_images, test_masks):
        super(MetricsCallback, self).__init__()
        self.train_images = train_images
        self.train_masks = train_masks
        self.test_images = test_images
        self.test_masks = test_masks

    def on_epoch_end(self, epoch, logs=None):
        print("\n============================================================================")
        print(f"Epoch {epoch + 1}")

        print("Train test : ")
        predicted_masks = self.model.predict(self.train_images)
        predicted_masks = np.argmax(predicted_masks, axis=-1)

        true_masks = self.train_masks.flatten()
        predicted_masks = predicted_masks.flatten()

        print("Pixel Accuracy:", accuracy_score(true_masks, predicted_masks))
        print("Precision:", precision_score(true_masks, predicted_masks, zero_division=0))
        print("Recall:", recall_score(true_masks, predicted_masks, zero_division=0))
        print("F1-score:", f1_score(true_masks, predicted_masks, zero_division=0))
        print("Mean IoU:", jaccard_score(true_masks, predicted_masks, average='macro'))
        print("Confusion Matrix:")
        print(confusion_matrix(true_masks, predicted_masks))

        print("\nTest test : ")
        predicted_masks = self.model.predict(self.test_images)
        predicted_masks = np.argmax(predicted_masks, axis=-1)

        true_masks = self.test_masks.flatten()
        predicted_masks = predicted_masks.flatten()

        print("Pixel Accuracy:", accuracy_score(true_masks, predicted_masks))
        print("Precision:", precision_score(true_masks, predicted_masks, zero_division=0))
        print("Recall:", recall_score(true_masks, predicted_masks, zero_division=0))
        print("F1-score:", f1_score(true_masks, predicted_masks, zero_division=0))
        print("Mean IoU:", jaccard_score(true_masks, predicted_masks, average='macro'))
        print("Confusion Matrix:")
        print(confusion_matrix(true_masks, predicted_masks))
        print("============================================================================")
        return
