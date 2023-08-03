from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib


# Find and return the best model and parameters for the Random Forest
def grindsearchlearn(param_grid, x, y, filename):
    rf_model = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1,
                               verbose=10)
    grid_search.fit(x, y)

    best_params = grid_search.best_params_

    best_rf_model = RandomForestClassifier(**best_params, random_state=42)
    best_rf_model.fit(x, y)
    joblib.dump(best_rf_model, 'output/' + filename)
    joblib.dump(best_params, 'output/' + 'parametri_' + filename)


# Return the predictions and the parameters used in the Random Forest model
def rftest(test, filename):
    rf_model = joblib.load('output/' + filename)
    param = joblib.load('output/' + 'parametri_' + filename)
    predictions = rf_model.predict(test)
    return predictions, param
