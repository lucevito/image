import tensorflow as tf
import numpy as np

from utility import print_save_metrics


def predict(model_name, test_images, path):
    """
    Perform image segmentation prediction using a saved TensorFlow model.

    Args:
        model_name (str): Name of the saved model file.
        test_images (numpy.ndarray): Input images for prediction.
        path (str): Directory path where the model is saved.

    Returns:
        predicted_masks (numpy.ndarray): Predicted binary masks.

    """
    model = tf.keras.models.load_model(path + model_name)
    predicted_masks = model.predict(test_images)
    predicted_masks = (predicted_masks > 0.5).astype(np.uint8)
    return predicted_masks


def predict_save(model_name, dataset, param, test_images, test_masks, path):
    """
    Perform image segmentation prediction and save evaluation metrics.

    Args:
        model_name (str): Name of the saved model file.
        dataset (str): Name of the dataset.
        param (str): Additional parameter description.
        test_images (numpy.ndarray): Input images for prediction.
        test_masks (numpy.ndarray): Ground truth masks for evaluation.
        path (str): Directory path where the model is saved.

    Returns:
        None

    """
    pred_masks = predict(model_name, test_images, path)
    pred_masks = pred_masks.flatten()
    true_masks = test_masks.flatten()
    print_save_metrics(model_name, dataset, param, true_masks, pred_masks)
