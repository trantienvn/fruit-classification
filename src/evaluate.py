"""Đánh giá mô hình phân loại — tính metrics và xuất biểu đồ báo cáo.

Các metrics:
  - Accuracy, Precision, Recall, F1 (macro/weighted)
  - Confusion matrix
  - Classification report chi tiết theo class
  - ROC / AUC (one-vs-rest, khi model hỗ trợ predict_proba)
  - So sánh hiệu năng nhiều mô hình

Biểu đồ lưu vào reports/:
  - confusion_matrix.png
  - roc_curves.png
  - model_comparison.png
  - feature_importance.png (nếu mô hình hỗ trợ)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # không cần display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def evaluate_predictions(
    y_true: list[int],
    y_pred: list[int],
    class_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Tính metrics phân loại cơ bản.

    Trả về dict gồm accuracy, macro_f1, report và các chỉ số khác.
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_t, y_p)),
        "macro_f1": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_t, y_p, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_t, y_p, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_t, y_p, average="macro", zero_division=0)),
        "report": classification_report(
            y_t, y_p,
            target_names=list(class_names) if class_names else None,
            zero_division=0,
            output_dict=True,
        ),
    }
    return result


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> None:
    """Vẽ và lưu confusion matrix."""
    labels = [n.replace(" ", "\n") for n in class_names]
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, colorbar=False
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Tập Test", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu: {save_path}")


def _plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> dict[str, float]:
    """Vẽ ROC curves (OvR) và lưu. Trả về dict AUC theo class."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(10, 7))
    aucs: dict[str, float] = {}

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))  # type: ignore[attr-defined]
    for i, (name, color) in enumerate(zip(class_names, colors)):
        if np.sum(y_bin[:, i]) == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[name] = float(roc_auc)
        short = name.replace(" ", "\n")
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{short} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One-vs-Rest (Test)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu: {save_path}")
    return aucs


def _plot_model_comparison(
    all_results: dict[str, dict],
    save_path: Path,
) -> None:
    """Biểu đồ so sánh hiệu năng các mô hình trên val."""
    names = list(all_results.keys())
    accs = [v.get("accuracy", 0) for v in all_results.values()]
    f1s = [v.get("macro_f1", 0) for v in all_results.values()]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, f1s, width, label="Macro F1", color="#DD8452")

    ax.set_title("So sánh hiệu năng mô hình (Val)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", fontsize=7, padding=2)
    ax.bar_label(bars2, fmt="%.3f", fontsize=7, padding=2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu: {save_path}")


def _plot_feature_importance(
    model: object,
    feature_names: list[str],
    save_path: Path,
    top_n: int = 20,
) -> None:
    """Vẽ feature importance nếu mô hình hỗ trợ."""
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_  # type: ignore[union-attr]
    elif hasattr(model, "coef_"):
        coef = model.coef_  # type: ignore[union-attr]
        importance = np.abs(coef).mean(axis=0)

    if importance is None:
        return

    idx = np.argsort(importance)[-top_n:]
    top_names = [feature_names[i] for i in idx]
    top_vals = importance[idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_names, top_vals, color="#4C72B0")
    ax.set_title(f"Top {top_n} Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu: {save_path}")


def full_report(
    y_true: list[int],
    y_pred: list[int],
    class_names: tuple[str, ...] | list[str],
    model_name: str,
    all_val_results: dict[str, dict],
    save_dir: Path,
    model: object | None = None,
    X_test: np.ndarray | None = None,
    X_val: np.ndarray | None = None,
) -> None:
    """Tạo toàn bộ biểu đồ và in classification report."""
    from src.features import FEATURE_NAMES

    save_dir.mkdir(parents=True, exist_ok=True)
    class_list = list(class_names)
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    print("\n── Xuất biểu đồ báo cáo ──")

    # Confusion matrix
    _plot_confusion_matrix(y_t, y_p, class_list, save_dir / "confusion_matrix.png")

    # ROC curves (nếu model hỗ trợ predict_proba)
    X_eval = X_test if X_test is not None else X_val
    if model is not None and X_eval is not None and hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_eval)  # type: ignore[union-attr]
        aucs = _plot_roc_curves(
            np.array(y_true), y_score, class_list, save_dir / "roc_curves.png"
        )
        if aucs:
            print(f"  Macro AUC: {np.mean(list(aucs.values())):.4f}")

    # So sánh models
    _plot_model_comparison(all_val_results, save_dir / "model_comparison.png")

    # Feature importance
    if model is not None:
        _plot_feature_importance(model, FEATURE_NAMES, save_dir / "feature_importance.png")

    # In classification report
    report_str = classification_report(
        y_t, y_p, target_names=class_list, zero_division=0
    )
    print("\nClassification Report (Test):")
    print(report_str)

    report_path = save_dir / f"classification_report_{model_name}.txt"
    report_path.write_text(
        f"Model: {model_name}\n\n{report_str}", encoding="utf-8"
    )
    print(f"  Đã lưu: {report_path}")
