"""
Evaluation metrics implementation.
Corresponds to Section 2.5.3 and Tables 3-4 of the manuscript.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute classification metrics as reported in the manuscript.
    
    Args:
        y_true: Ground truth labels, shape (N,)
        y_pred: Predicted labels, shape (N,)
        y_prob: Predicted probabilities for the positive (malignant) class, shape (N,)
    
    Returns:
        dict: Dictionary containing accuracy, sensitivity, specificity,
              precision, f1_score, and auc (if y_prob is provided).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred, zero_division=0),  # Recall for malignant class
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['auc'] = None
    
    return metrics


def print_confusion_matrix(y_true, y_pred, class_names=('Benign', 'Malignant')):
    """
    Print confusion matrix in the format of Table 3 / Figure 4.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:")
    print(f"                 Pred {class_names[0]}   Pred {class_names[1]}")
    print(f"True {class_names[0]:10}   {tn:4}       {fp:4}")
    print(f"True {class_names[1]:10}   {fn:4}       {tp:4}")
    return cm
