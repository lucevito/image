from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
import pandas as pd
import os
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


# Print the metrics to the screen and save them in an Excel workbook named risultati_modelli.xlsx
def print_save_metrics(model_name, dataset_name, param, y_true, y_pred):
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
