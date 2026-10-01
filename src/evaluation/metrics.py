"""Performance metrics calculation for Motor Imagery EEG classification.
Computes Accuracy, Precision, Recall, F1, Confusion Matrix, and ROC-AUC.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 4,
    class_names: Tuple[str, ...] = ("Left Fist", "Right Fist", "Both Fists", "Both Feet"),
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Per-class accuracy
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    # Normalize confusion matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        conf_mat_norm = conf_mat.astype(np.float64) / conf_mat.sum(axis=1, keepdims=True)
        conf_mat_norm = np.nan_to_num(conf_mat_norm)

    per_class_acc = conf_mat_norm.diagonal()

    # ROC Curves & AUC (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    fpr: Dict[int, np.ndarray] = {}
    tpr: Dict[int, np.ndarray] = {}
    roc_auc: Dict[int, float] = {}

    for i in range(num_classes):
        if y_true_bin.shape[1] > 1:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[i], tpr[i] = np.array([0, 1]), np.array([0, 1])
            roc_auc[i] = 0.5

    macro_auc = float(np.mean(list(roc_auc.values())))

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "per_class_accuracy": per_class_acc.tolist(),
        "per_class_precision": per_class_p.tolist(),
        "per_class_recall": per_class_r.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": conf_mat,
        "confusion_matrix_norm": conf_mat_norm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "macro_auc": macro_auc,
        "class_names": list(class_names),
    }
