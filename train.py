from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib


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
