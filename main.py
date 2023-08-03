import os

from dataset import loaddataset, reduce_forsample
from learn import grindsearchlearn, rftest
from metric import print_save_metrics

# Train path
train_path = 'Immagini_satellitari/Train'
# Test path
test_path = 'Immagini_satellitari/Test/'
# Model name
model_name = "rf_GridSearch_model.h"
# First part of the string containing the used parameters
param = 'rf GridSearch'

# Parameters for GridSearch
param_grid = {
    "class_weight": [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}, "balanced"],
    "max_depth": [7, 8, 9, 10],
    "max_samples": [0.8, 0.9, 1.0],
    'criterion': ['entropy', 'gini', 'log_loss'],
    "max_features": ["sqrt", "log2"]
}

outputdir = "/output"
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

print("Loading train...")
trainX, trainY = loaddataset(train_path)
print("Loading test...")
testX, testY = loaddataset(test_path)
print("Finding best parameters... ")
grindsearchlearn(param_grid, trainX, trainY, model_name)
print("GridSearch complete")


print("Starting to predict test values...")
test_predictions, best_params = rftest(testX, model_name)
param = param + str(best_params)
print("TEST : ")
print_save_metrics(model_name, 'Test Set', param, testY, test_predictions)

print("Starting to predict train values...")
train_predictions, best_params = rftest(trainX, model_name)
print("TRAIN : ")
print_save_metrics(model_name, 'Train Set', param, trainY, train_predictions)

model_name_sample = "rf_GridSearch_model_sample_n_"
for i in range(1, 11):
    print("Reducing in sample train for the number of element of the class 1 * " + str(i))
    x, y = reduce_forsample(trainX, trainY, i)
    m_name = model_name_sample + str(i) + ".h"
    print("Finding best parameters... ")
    grindsearchlearn(param_grid, x, y, m_name)
    print("Grindsearch complete")
    param_sample = 'rf GridSearch sample n ' + str(i) + ' '
    print("Starting to predict test values...")
    test_predictions, best_params = rftest(testX, m_name)
    param_sample = param_sample + str(best_params)
    print("TEST : ")
    print_save_metrics(m_name, 'Test Set', param_sample, testY, test_predictions)

    print("Starting to predict train values...")
    train_predictions, best_params = rftest(trainX, m_name)
    print("TRAIN : ")
    print_save_metrics(m_name, 'Train Set', param_sample, trainY, train_predictions)
