from typing import Literal

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from writing_assistance import FormalityDetector


def generate_predictions(
    df: pd.DataFrame, detector_model: Literal['deberta', 'xlm_roberta', 'gpt']
) -> tuple[np.ndarray, np.ndarray]:
    """Generates formality predictions for a given dataset using the specified model."""
    y_true = df['avg_score'].values
    sentences = df['sentence'].tolist()

    predictions = FormalityDetector.predict(detector_model, sentences)
    y_pred = np.array([pred['formal'] for pred in predictions])

    return y_true, y_pred


def evaluate_formality_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Evaluates formality predictions using RMSE, MAE, and R2 metrics."""
    metrics = ['RMSE', 'MAE', 'R2']

    results = {metric: [] for metric in metrics}

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    results['RMSE'].append(rmse)
    results['MAE'].append(mae)
    results['R2'].append(r2)

    metrics_df = pd.DataFrame(results, index=['All Data'])

    return metrics_df


def generate_confusion_matrix_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Converts continuous values to binary (0 or 1) and computes confusion matrix, precision, and recall.
    """
    y_pred_discrete = (y_pred > 0.5).astype(int)
    y_true_discrete = (y_true > 0.5).astype(int)
    cm = confusion_matrix(y_true_discrete, y_pred_discrete)

    precision = precision_score(y_true_discrete, y_pred_discrete)
    recall = recall_score(y_true_discrete, y_pred_discrete)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return precision, recall


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plots the ROC curve and computes the AUC score.
    """
    y_true = (y_true > 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


def plot_scatter_true_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Plots a scatter plot of true vs predicted formality scores."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.6, edgecolors='k', s=50)
    plt.title('True vs Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--')
    plt.show()


def plot_error_histogram(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Plots a histogram of average prediction errors, binned by true values."""
    bins = np.arange(0, 1.1, 0.1)
    bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins) - 1)]

    errors = np.abs(y_true - y_pred)

    df = pd.DataFrame({'y_true': y_true, 'error': errors})

    df['bin'] = pd.cut(df['y_true'], bins=bins, labels=bin_labels, include_lowest=True)
    error_means = df.groupby('bin', observed=False)['error'].mean()

    plt.figure(figsize=(10, 6))
    error_means.plot(kind='bar', color='blue', alpha=0.7)
    plt.xlabel('True Value Range (Binned)')
    plt.ylabel('MAE')
    plt.title('MAE per y_true Bin')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()
