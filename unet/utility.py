from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
import pandas as pd
import os
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


def create_output_directory(output_path):
    """
    Create the output directory if it doesn't exist.

    Parameters:
        output_path (str): The path to the output directory.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)


def create_subdirectories(output_path, subdirectories):
    """
    Create subdirectories within the output directory if they don't exist.

    Parameters:
        output_path (str): The path to the output directory.
        subdirectories (list): A list of subdirectory names to create.
    """
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(output_path, subdirectory)
        if not os.path.exists(subdirectory_path):
            os.makedirs(subdirectory_path)


def print_save_metrics(model_name, dataset_name, param, y_true, y_pred):
    """
    Calculates and prints various classification metrics for a given model's predictions and true labels.
    The function saves the results to an Excel file for further analysis and comparison.

    Parameters:
        model_name (str): A string that represents the name of the machine learning model used for predictions.
        dataset_name (str): A string that represents the name of the dataset used for evaluation.
        param (str or dict): A string or dictionary that represents the configuration parameters used in the model (e.g., hyperparameters, settings).
        y_true (numpy array or list): A 1-dimensional numpy array or list containing the true labels.
        y_pred (numpy array or list): A 1-dimensional numpy array or list containing the predicted labels generated by the model.

    Metrics Computed and Printed:
        The following classification metrics are calculated and printed for both the positive and negative classes:
        - True Negative (TN): The number of true negative predictions.
        - False Negative (FN): The number of false negative predictions.
        - False Positive (FP): The number of false positive predictions.
        - True Positive (TP): The number of true positive predictions.
        - Precision (Negative Class): Precision (also called Positive Predictive Value) for the negative class.
        - Recall (Negative Class): Recall (also called Sensitivity or True Positive Rate) for the negative class.
        - F-score (Negative Class): F1-score for the negative class, which is the harmonic mean of precision and recall.
        - Precision (Positive Class): Precision for the positive class.
        - Recall (Positive Class): Recall for the positive class.
        - F-score (Positive Class): F1-score for the positive class, which is the harmonic mean of precision and recall.
        - Average Accuracy: The average of normalized and unnormalized accuracy scores.
        - Overall Accuracy: The normalized accuracy score.
        - G-Mean: Geometric Mean Score, which is a balanced metric that considers both sensitivity and specificity.
        - AUC (Area Under the Curve): The area under the Receiver Operating Characteristic (ROC) curve.

    Output:
        The function will print the computed metrics to the console.

    Excel File Saving:
        The function saves the computed metrics along with the model name, dataset name, and configuration parameters
        to an Excel file named 'output/risultati_modelli.xlsx'. The function creates or appends to the existing file based on its presence.

        The Excel file has the following columns:
        - Modello (Model Name)
        - Dataset (Dataset Name)
        - Parametri della configurazione (Configuration Parameters)
        - True Negative
        - False Negative
        - False Positive
        - True Positive
        - Precision Negative
        - Recall Negative
        - Fscore Negative
        - Precision Positive
        - Recall Positive
        - Fscore Positive
        - Average Accuracy
        - Overall Accuracy
        - GMean
        - AUC

    """
    file_name = 'output/risultati_modelli.xlsx'
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision_negative = precision_score(y_true, y_pred, pos_label=0)
    recall_negative = recall_score(y_true, y_pred, pos_label=0)
    fscore_negative = f1_score(y_true, y_pred, pos_label=0)
    precision_positive = precision_score(y_true, y_pred, pos_label=1)
    recall_positive = recall_score(y_true, y_pred, pos_label=1)
    fscore_positive = f1_score(y_true, y_pred, pos_label=1)
    average_accuracy = (accuracy_score(y_true, y_pred) +
                        accuracy_score(y_true, y_pred, normalize=False)) / 2
    overall_accuracy = accuracy_score(y_true, y_pred)
    gmean = geometric_mean_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    print("True Negative (TN):", tn)
    print("False Negative (FN):", fn)
    print("False Positive (FP):", fp)
    print("True Positive (TP):", tp)
    print("Precision (Negative Class):", precision_negative)
    print("Recall (Negative Class):", recall_negative)
    print("F-score (Negative Class):", fscore_negative)
    print("Precision (Positive Class):", precision_positive)
    print("Recall (Positive Class):", recall_positive)
    print("F-score (Positive Class):", fscore_positive)
    print("Average Accuracy:", average_accuracy)
    print("Overall Accuracy:", overall_accuracy)
    print("G-Mean:", gmean)
    print("AUC (Area Under the Curve):", roc_auc)
    print("\n")

    results = [
        {
            'Modello': model_name,
            'Dataset': dataset_name,
            'Parametri della configurazione': param,
            'True Negative': tn,
            'False Negative': fn,
            'False Positive': fp,
            'True Positive': tp,
            'Precision Negative': precision_negative,
            'Recall Negative': recall_negative,
            'Fscore Negative': fscore_negative,
            'Precision Positive': precision_positive,
            'Recall Positive': recall_positive,
            'Fscore Positive': fscore_positive,
            'Average Accuracy': average_accuracy,
            'Overall Accuracy': overall_accuracy,
            'GMean': gmean,
            'AUC': roc_auc,
        },
    ]

    if os.path.exists(file_name):
        existing_df = pd.read_excel(file_name)
        df = pd.concat([existing_df, pd.DataFrame(results)])
    else:
        df = pd.DataFrame(results)
    wb = Workbook()
    ws = wb.active
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)
    wb.save(file_name)
