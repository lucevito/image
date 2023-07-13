import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, jaccard_score, confusion_matrix, f1_score
import tensorflow as tf
# valori in input per create_unet() sono presenti in config.py
def create_unet(input_shape, num_classes, kernel, pool_size, encoder_filters, decoder_filters):
    # (32,32,10)
    inputs = tf.keras.Input(input_shape)
    x = inputs

    conv_layers = []
    pooling_layers = []
    for filters in encoder_filters:
        conv = tf.keras.layers.Conv2D(filters, kernel, activation='relu', padding='same')(x)
        conv = tf.keras.layers.Conv2D(filters, kernel, activation='relu', padding='same')(conv)
        pool = tf.keras.layers.MaxPooling2D(pool_size)(conv)
        conv_layers.append(conv)
        pooling_layers.append(pool)
        x = pool

    bottleneck = tf.keras.layers.Conv2D(encoder_filters[-1], kernel,
                                        activation='relu', padding='same')(pooling_layers[-1])
    bottleneck = tf.keras.layers.Conv2D(encoder_filters[-1], kernel,
                                        activation='relu', padding='same')(bottleneck)

    for filters in reversed(decoder_filters):
        up = tf.keras.layers.UpSampling2D(pool_size)(bottleneck)
        concat = tf.keras.layers.Concatenate()([up, conv_layers.pop()])
        conv = tf.keras.layers.Conv2D(filters, kernel, activation='relu', padding='same')(concat)
        conv = tf.keras.layers.Conv2D(filters, kernel, activation='relu', padding='same')(conv)
        bottleneck = conv

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid')(bottleneck)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
