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

import csv
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
from sklearn.model_selection import StratifiedKFold, learning_curve


def _macro_f1_scorer(model: object, X: np.ndarray, y_true: np.ndarray) -> float:
    """Score predictions without requiring a full scikit-learn estimator API."""
    y_pred = model.predict(X)  # type: ignore[union-attr]
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


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


def _save_model_comparison_csv(
    all_results: dict[str, dict],
    save_path: Path,
) -> None:
    """Lưu metrics validation và thời gian đo được cho từng mô hình."""
    fields = (
        "model", "accuracy", "macro_f1", "weighted_f1", "macro_precision",
        "macro_recall", "train_time_s", "infer_time_ms", "tune_time_s",
    )
    with save_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for name, metrics in all_results.items():
            writer.writerow({"model": name, **{key: metrics.get(key, "") for key in fields[1:]}})
    print(f"  Đã lưu: {save_path}")


def _save_ablation_report(results: dict[str, dict], save_dir: Path) -> None:
    """Lưu bảng và biểu đồ so sánh các nhóm đặc trưng trên validation."""
    csv_path = save_dir / "ablation_comparison.csv"
    fields = ("feature_set", "best_model", "accuracy", "macro_f1")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"feature_set": name, **values} for name, values in results.items())

    names = list(results)
    scores = [results[name]["macro_f1"] for name in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, scores, color=("#4C72B0", "#55A868", "#C44E52"))
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation macro F1")
    ax.set_title("Ablation đặc trưng")
    fig.tight_layout()
    fig.savefig(save_dir / "ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu: {csv_path} và ablation_comparison.png")


def _save_per_fruit_report(results: dict[str, dict], save_dir: Path) -> None:
    """Lưu global-vs-per-fruit metrics đã chọn trên val và đo trên test."""
    fields = ("scope", "fruit", "model", "val_macro_f1", "test_accuracy", "test_macro_f1")
    with (save_dir / "per_fruit_comparison.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def _plot_learning_curve(
    model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_path: Path,
) -> tuple[float, float]:
    """Vẽ learning curve 5-fold chỉ từ tập train."""
    sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_macro",
        train_sizes=np.linspace(0.5, 1.0, 5),
        shuffle=True,
        random_state=42,
        n_jobs=1,
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_mean, marker="o", label="Train macro F1")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    ax.plot(sizes, val_mean, marker="o", label="CV macro F1")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
    ax.set_xlabel("Số mẫu train")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.02)
    ax.set_title("Learning curve (5-fold trên train)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    with save_path.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(("n_train", "train_macro_f1", "train_std", "cv_macro_f1", "cv_std"))
        writer.writerows(zip(sizes, train_mean, train_std, val_mean, val_std))
    print(f"  Đã lưu: {save_path}")
    return float(train_mean[-1]), float(val_mean[-1])


def _save_misclassified_images(
    images: list[np.ndarray],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_dir: Path,
) -> None:
    """Lưu ảnh test dự đoán sai và bảng nhãn thật/dự đoán."""
    from PIL import Image

    error_dir = save_dir / "misclassified"
    error_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pair_counts: dict[tuple[str, str], int] = {}
    for index in np.flatnonzero(y_true != y_pred):
        true_name, pred_name = class_names[int(y_true[index])], class_names[int(y_pred[index])]
        safe_true = true_name.replace(" ", "_")
        safe_pred = pred_name.replace(" ", "_")
        Image.fromarray(images[index]).save(
            error_dir / f"{index:04d}_{safe_true}_pred_{safe_pred}.png"
        )
        rows.append({"index": int(index), "true_label": true_name, "predicted_label": pred_name})
        pair = (true_name, pred_name)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    with (save_dir / "misclassified.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=("index", "true_label", "predicted_label"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Đã lưu {len(rows)} ảnh dự đoán sai: {error_dir}")
    if pair_counts:
        print("  Cặp lớp nhầm nhiều nhất:", sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)[:5])


def _save_classification_metrics_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> None:
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0, output_dict=True,
    )
    fields = ("label", "precision", "recall", "f1-score", "support")
    with save_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                writer.writerow({"label": label, **metrics})


def _write_summary(
    save_path: Path,
    model_name: str,
    val_metrics: dict,
    test_metrics: dict,
    learning_scores: tuple[float, float] | None,
    misclassification_pairs: list[tuple[str, str, int]],
) -> None:
    lines = [
        "# Báo cáo phân loại trái cây",
        "",
        f"- Mô hình được chọn bằng validation: **{model_name}**.",
        f"- Validation: accuracy={val_metrics.get('accuracy', float('nan')):.4f}, macro F1={val_metrics.get('macro_f1', float('nan')):.4f}.",
        f"- Test (đánh giá cuối): accuracy={test_metrics['accuracy']:.4f}, macro F1={test_metrics['macro_f1']:.4f}.",
        "",
        "## Diễn giải",
    ]
    if learning_scores is not None:
        train_score, cv_score = learning_scores
        gap = train_score - cv_score
        if gap > 0.1:
            diagnosis = "Có dấu hiệu overfit theo learning curve (khoảng cách train-CV lớn)."
        elif train_score < 0.6 and cv_score < 0.6:
            diagnosis = "Có dấu hiệu underfit hoặc feature chưa đủ phân biệt (điểm train và CV đều thấp)."
        else:
            diagnosis = "Learning curve không cho thấy khoảng cách train-CV lớn theo ngưỡng heuristic 0.10."
        lines.extend([
            f"- Learning curve ở kích thước train lớn nhất: train F1={train_score:.4f}, CV F1={cv_score:.4f}.",
            f"- {diagnosis}",
        ])
    lines.extend([
        "",
        "## Cặp lớp nhầm",
    ])
    if misclassification_pairs:
        lines.extend(
            f"- {true_name} → {predicted_name}: {count} ảnh."
            for true_name, predicted_name, count in misclassification_pairs
        )
    else:
        lines.append("- Không có ảnh test dự đoán sai.")
    lines.extend([
        "- Ánh sáng, nền và góc chụp là giả thuyết cần kiểm tra trực quan trên `misclassified/`; báo cáo không tự suy diễn nguyên nhân.",
        "",
        "## Hạn chế và hướng cải tiến",
        "- Dataset và split hiện có còn giới hạn; kết quả không đại diện cho mọi điều kiện chụp.",
        "- Otsu/morphology có thể tách mask kém khi nền phức tạp; cần kiểm tra `mask_samples/` và ảnh trong `misclassified/`.",
        "- Bổ sung ảnh chụp thực tế đa dạng ánh sáng/góc/nền và đánh giá độc lập trước khi dùng ngoài phạm vi dataset.",
    ])
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_feature_importance(
    model: object,
    feature_names: list[str],
    save_path: Path,
    X_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    top_n: int = 20,
) -> bool:
    """Vẽ built-in importance hoặc permutation importance trên validation."""
    from sklearn.pipeline import Pipeline

    target_model = model
    eval_X = X_eval
    target_names = feature_names
    if isinstance(model, Pipeline) and "model" in model.named_steps:
        target_model = model.named_steps["model"]
        pca_step = model.named_steps.get("pca")
        if eval_X is not None and pca_step is not None and pca_step != "passthrough":
            eval_X = model[:-1].transform(eval_X)
            target_names = [f"PC_{index + 1}" for index in range(eval_X.shape[1])]

    importance = None
    importance_std = None
    if hasattr(target_model, "feature_importances_"):
        importance = target_model.feature_importances_  # type: ignore[union-attr]
    elif hasattr(target_model, "coef_"):
        coef = target_model.coef_  # type: ignore[union-attr]
        importance = np.abs(coef).mean(axis=0)
    elif eval_X is not None and y_eval is not None:
        if len(target_names) > 500:
            return False
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            target_model,
            eval_X,
            y_eval,
            scoring=_macro_f1_scorer,
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )
        importance = result.importances_mean
        importance_std = result.importances_std

    if importance is None:
        return False
    importance = np.asarray(importance).reshape(-1)
    if len(importance) != len(target_names) or not np.all(np.isfinite(importance)):
        raise ValueError("Số lượng hoặc giá trị feature importance không hợp lệ.")

    idx = np.argsort(importance)[-min(top_n, len(importance)):]
    top_names = [target_names[i] for i in idx]
    top_vals = importance[idx]
    errors = importance_std[idx] if importance_std is not None else None
    displayed_n = len(idx)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_names, top_vals, xerr=errors, color="#4C72B0")
    ax.set_title(f"Top {displayed_n} Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel(
        "Permutation importance (Macro F1 decrease)"
        if importance_std is not None
        else "Importance"
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu: {save_path}")
    return True


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
    y_val: list[int] | None = None,
    X_train: np.ndarray | None = None,
    y_train: list[int] | None = None,
    test_images: list[np.ndarray] | None = None,
    ablation_results: dict[str, dict] | None = None,
    per_fruit_results: list[dict] | None = None,
    selected_val_metrics: dict | None = None,
    feature_names: list[str] | None = None,
) -> None:
    """Tạo toàn bộ biểu đồ và in classification report."""
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
            with (save_dir / "roc_auc.csv").open("w", newline="", encoding="utf-8-sig") as output:
                writer = csv.writer(output)
                writer.writerow(("class", "auc_ovr"))
                writer.writerows(aucs.items())

    # So sánh models
    _plot_model_comparison(all_val_results, save_dir / "model_comparison.png")
    _save_model_comparison_csv(all_val_results, save_dir / "model_comparison.csv")
    if ablation_results:
        _save_ablation_report(ablation_results, save_dir)
    if per_fruit_results:
        _save_per_fruit_report(per_fruit_results, save_dir)

    learning_scores = None
    if model is not None and X_train is not None and y_train is not None:
        learning_scores = _plot_learning_curve(
            model, X_train, np.asarray(y_train), save_dir / "learning_curve.png"
        )

    if test_images is not None:
        _save_misclassified_images(test_images, y_t, y_p, class_list, save_dir)

    # Feature importance: prefer validation data for model-agnostic estimates.
    if model is not None:
        resolved_names = feature_names
        if resolved_names is None:
            from src.features import FEATURE_NAMES
            resolved_names = FEATURE_NAMES
        if X_val is not None and X_val.shape[1] != len(resolved_names):
            resolved_names = [f"component_{index + 1}" for index in range(X_val.shape[1])]
        generated = _plot_feature_importance(
            model,
            resolved_names,
            save_dir / "feature_importance.png",
            X_eval=X_val,
            y_eval=np.asarray(y_val) if y_val is not None else None,
        )
        if not generated:
            print("  Không tạo feature_importance.png: model không hỗ trợ importance.")

    # In classification report
    _save_classification_metrics_csv(
        y_t, y_p, class_list, save_dir / "classification_metrics.csv"
    )
    report_str = classification_report(
        y_t, y_p, target_names=class_list, zero_division=0
    )
    print("\nClassification Report (Test):")
    print(report_str)

    report_path = save_dir / f"classification_report_{model_name}.txt"
    report_path.write_text(
        f"Model: {model_name}\n\n{report_str}", encoding="utf-8"
    )
    selected_val = selected_val_metrics or all_val_results.get(
        f"{model_name}_tuned", all_val_results.get(model_name, {})
    )
    test_metrics = evaluate_predictions(y_true, y_pred, class_names=class_list)
    pair_counts: dict[tuple[str, str], int] = {}
    for true_id, predicted_id in zip(y_t, y_p):
        if true_id != predicted_id:
            pair = (class_list[int(true_id)], class_list[int(predicted_id)])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    _write_summary(
        save_dir / "summary.md", model_name, selected_val, test_metrics,
        learning_scores,
        [(true_name, predicted_name, count) for (true_name, predicted_name), count in
         sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)[:5]],
    )
    print(f"  Đã lưu: {report_path}")
