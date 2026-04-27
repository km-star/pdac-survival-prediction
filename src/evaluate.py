"""
evaluate.py
===========
Evaluation metrics, ROC curves, survival plots, and model comparison table.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    classification_report, accuracy_score, roc_auc_score
)
import logging

logger = logging.getLogger(__name__)


def full_report(y_true, y_pred, y_prob=None, model_name: str = "Model"):
    """Print full classification report with optional AUC."""
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    if y_prob is not None:
        print(f"ROC-AUC:  {roc_auc_score(y_true, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Short-survivor", "Long-survivor"]))


def plot_roc_curves(models_results: dict, save_path: str = "results/roc_curves.png"):
    """
    Plot ROC curves for multiple models on the same axes.

    models_results: {model_name: {"fpr": [...], "tpr": [...], "auc": float}}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, r in models_results.items():
        ax.plot(r["fpr"], r["tpr"], label=f"{name} (AUC={r['auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — PDAC 2-Year Survival Prediction", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"ROC curves saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Ensemble",
                          save_path: str = "results/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Short", "Long"], yticklabels=["Short", "Long"])
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def model_comparison_table(results: list[dict], save_path: str = "results/model_comparison.csv"):
    """
    Build and save a model comparison table.

    results: list of dicts with keys: model, accuracy, std_acc, auc, std_auc, notes
    """
    df = pd.DataFrame(results)
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    df["accuracy"] = df["accuracy"].map("{:.3f}".format)
    df["auc"] = df["auc"].map("{:.3f}".format)
    df.to_csv(save_path, index=False)
    logger.info(f"Model comparison table saved to {save_path}")
    print(df.to_string(index=False))
    return df


def benchmark_comparison():
    """Print comparison against Nature Cancer (2024) benchmark."""
    our_acc = 0.841
    benchmark_acc = 0.850
    gap = benchmark_acc - our_acc
    print(f"\nOur best result:        ACC = {our_acc:.3f}")
    print(f"Nature Cancer benchmark: ACC = {benchmark_acc:.3f}")
    print(f"Gap:                     {gap:.3f} ({gap/benchmark_acc*100:.1f}%)")
    print("\nNote: gap attributable to smaller TCGA-PAAD cohort vs. "
          "multi-institutional dataset used in the original paper.")
