from dataset import load_dataset, create_smote_dataset, create_sample_dataset, create_samplesmote_dataset
from learn import gr_search, sample_grindseach_learn, test, sample_test
from utility import create_output_directory, create_subdirectories

# Parameters for GridSearch
param_grid = {
    "class_weight": [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}, "balanced"],
    "max_depth": [7, 8, 9, 10],
    "max_samples": [0.8, 0.9, 1.0],
    'criterion': ['entropy', 'gini', 'log_loss'],
    "max_features": ["sqrt", "log2"]
}

# Train path
train_path = 'Immagini_satellitari/Train'
# Test path
test_path = 'Immagini_satellitari/Test/'

# Create the main output directory
output_path = 'output'
create_output_directory(output_path)

# Create subdirectories 'sample', 'smote', and 'samplesmote'
subdirectories = ['sample', 'smote', 'samplesmote']
create_subdirectories(output_path, subdirectories)

print("Loading train...")
trainX, trainY = load_dataset(train_path)
print("Loading test...")
testX, testY = load_dataset(test_path)

model_name = "rf_GridSearch_model.h"
path = 'output/'
gr_search(param_grid, trainX, trainY, model_name, path)
test(trainX, trainY, testX, testY, model_name, path)

model_name_sample = "sample_model_"
path = 'output/sample/'
create_sample_dataset(trainX, trainY, model_name_sample, path, n_values=range(1, 11))
sample_grindseach_learn(param_grid, model_name_sample, path, start=1)
sample_test(model_name_sample, trainX, trainY, testX, testY, path)

model_name_smote = "smote_model_"
path = 'output/smote/'
create_smote_dataset(trainX, trainY, model_name_smote, path, n_values=range(2, 11))
sample_grindseach_learn(param_grid, model_name_smote, path, start=2)
sample_test(model_name_smote, trainX, trainY, testX, testY, path)

model_name_smote = "samplesmote_model_"
path = 'output/samplesmote/'
create_samplesmote_dataset(trainX, trainY, model_name_smote, path, n_values=range(2, 11))
sample_grindseach_learn(param_grid, model_name_smote, path, start=2)
sample_test(model_name_smote, trainX, trainY, testX, testY, path)
