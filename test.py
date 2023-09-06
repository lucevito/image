import joblib
from dataset import load_model_prefix
from utility import print_save_metrics
import tensorflow as tf
import numpy as np


def rftest(test_x, filename, path):
    """
    Load a RandomForestClassifier model and predict labels for test data.

    Parameters:
        test_x (numpy array): Input features for testing.
        filename (str): The filename of the RandomForestClassifier model to load.
        path (str): The path where the model is saved.

    Returns:
        numpy array: Predicted labels for the test data.
    """
    rf_model = joblib.load(path + filename)
    predictions = rf_model.predict(test_x)
    return predictions


def sample_test(prefix, train_x, train_y, test_x, test_y, path):
    """
    Test multiple RandomForestClassifier models with a common filename prefix on test and train datasets.

    Parameters:
        prefix (str): The common filename prefix of the models to test.
        train_x (numpy array): Input features for the training set.
        train_y (numpy array): Target labels for the training set.
        test_x (numpy array): Input features for the test set.
        test_y (numpy array): Target labels for the test set.
        path (str): The path where the models and related files are stored.
    """
    models, file_name = load_model_prefix(path, prefix)
    for model, name in zip(models, file_name):
        test(train_x, train_y, test_x, test_y, name, path)


def test(train_x, train_y, test_x, test_y, model_name, path):
    """
    Test a RandomForestClassifier model on both train and test datasets and print/save the metrics.

    Parameters:
        train_x (numpy array): Input features for the training set.
        train_y (numpy array): Target labels for the training set.
        test_x (numpy array): Input features for the test set.
        test_y (numpy array): Target labels for the test set.
        model_name (str): The filename of the RandomForestClassifier model to test.
        path (str): The path where the model is stored and related files will be saved.
    """
    param = str(joblib.load(path + 'parametri_' + model_name))

    print("Starting to predict test values...")
    test_predictions = rftest(test_x, model_name, path)
    print("TEST : ")
    print_save_metrics(model_name, 'Test Set', param, test_y, test_predictions)

    print("Starting to predict train values...")
    train_predictions = rftest(train_x, model_name, path)
    print("TRAIN : ")
    print_save_metrics(model_name, 'Train Set', param, train_y, train_predictions)


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


def unet_predict_save(model_name, dataset, param, test_images, test_masks, path):
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
