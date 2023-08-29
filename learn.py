from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
from dataset import load_model_prefix
from utility import print_save_metrics


def gr_search(param_grid, x, y, filename, path):
    """
    Perform grid search with cross-validation to find the best hyperparameters for the RandomForestClassifier.
    Save the model and the best hyperparameters in files

    Parameters:
        param_grid (dict): A dictionary containing the hyperparameter grid to search.
        x (numpy array): Input features for training.
        y (numpy array): Target labels for training.
        filename (str): The filename to save the best model obtained from grid search.
        path (str): The path where the model and related files will be saved.
    """
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1,
                               verbose=10)
    print("Finding best parameters... ")
    grid_search.fit(x, y)
    print("GridSearch complete")
    joblib.dump(grid_search.best_estimator_, path + filename)
    joblib.dump(grid_search.best_params_, path + 'parametri_' + filename)


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


def sample_grindseach_learn(param_grid, filename, path, start=1):
    """
    Perform grid search with cross-validation for multiple datasets and save the best models.

    Parameters:
        param_grid (dict): A dictionary containing the hyperparameter grid to search.
        filename (str): The common filename prefix of datasets.
        path (str): The path where the datasets are stored.
        start (int): The starting index to append to the filename when saving models. Default is 1.
    """
    datasets = joblib.load(path + 'Dataset_' + filename)
    for dataset_name, (x, y) in datasets.items():
        model = filename + str(start) + '.h'
        gr_search(param_grid, x, y, model, path)
        start = start + 1


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
