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
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # không cần display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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


def _plot_flow_diagram(
    title: str,
    steps: tuple[tuple[str, str], ...],
    save_path: Path,
) -> None:
    """Vẽ lưu đồ dọc từ các bước và mô tả ngắn."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    centers = np.linspace(0.91, 0.09, len(steps))
    box_width, box_height = 0.86, 0.095

    for upper, lower in zip(centers, centers[1:]):
        ax.add_patch(FancyArrowPatch(
            (0.5, upper - box_height / 2),
            (0.5, lower + box_height / 2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            color="#718477",
        ))

    for index, ((title, description), center) in enumerate(zip(steps, centers)):
        box = FancyBboxPatch(
            (0.5 - box_width / 2, center - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor="#edf3ea" if index % 2 == 0 else "#f7f8f4",
            edgecolor="#78917c",
            linewidth=1.1,
        )
        ax.add_patch(box)
        ax.text(
            0.5,
            center,
            f"{title}\n{description}",
            ha="center",
            va="center",
            fontsize=9,
            color="#26372e",
        )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_training_pipeline(save_path: Path, model_count: int, tuned_count: int) -> None:
    """Vẽ sơ đồ đúng theo luồng xử lý dữ liệu và huấn luyện hiện hành."""
    steps = (
        ("Dữ liệu có nhãn", "Dùng các split train / validation / test đã chuẩn bị"),
        ("Tiền xử lý ảnh", "RGB → resize → vector pixel chuẩn hóa [0, 1]"),
        ("So sánh mô hình", f"Fit {model_count} estimator trên train, chọn bằng macro F1 validation"),
        ("Tinh chỉnh", f"Top {tuned_count} mô hình · Stratified 5-fold CV trên train"),
        ("Tăng cường và fit cuối", "Chỉ tăng cường ảnh train; không thay đổi validation / test"),
        ("Đánh giá theo loại quả", "Chọn và fit mô hình riêng theo từng loại bằng train / validation"),
        ("Đánh giá độc lập", "Đo kết quả trên test sau khi đã chọn mô hình"),
        ("Lưu kết quả", "Model, metadata và biểu đồ dựa trên dữ liệu lần chạy"),
    )
    _plot_flow_diagram("Quy trình huấn luyện và đánh giá", steps, save_path)


def _plot_inference_pipeline(save_path: Path) -> None:
    """Vẽ luồng triển khai từ ảnh tải lên đến kết quả trả về trình duyệt."""
    steps = (
        ("Người dùng chọn ảnh", "Trình duyệt gửi file đến API /api/predict"),
        ("Xác thực ảnh", "Kiểm tra nội dung, định dạng và giới hạn dung lượng"),
        ("Tiền xử lý inference", "Đọc RGB, resize và tạo vector pixel chuẩn hóa"),
        ("Phân loại", "Pipeline đã nạp: scaler → PCA tùy chọn → classifier"),
        ("Lưu lịch sử", "Ghi nhãn và metadata; không lưu ảnh tải lên"),
        ("Trả kết quả", "API JSON trả nhãn, độ tin cậy và thời gian xử lý"),
    )
    _plot_flow_diagram("Luồng triển khai nhận diện ảnh", steps, save_path)


def _plot_dataset_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> None:
    """Vẽ số mẫu từng lớp trong các split thực tế."""
    n_classes = len(class_names)
    distributions = []
    for split_name, labels in (("Train", y_train), ("Validation", y_val), ("Test", y_test)):
        class_ids = np.asarray(labels, dtype=int)
        if class_ids.ndim != 1 or np.any(class_ids < 0) or np.any(class_ids >= n_classes):
            raise ValueError(f"Nhãn trong split {split_name} nằm ngoài class_names.")
        distributions.append(np.bincount(class_ids, minlength=n_classes))

    positions = np.arange(n_classes)
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, (name, counts, color) in enumerate(zip(
        ("Train", "Validation", "Test"),
        distributions,
        ("#4C72B0", "#55A868", "#DD8452"),
    )):
        ax.bar(positions + (index - 1) * width, counts, width, label=name, color=color)

    ax.set_xticks(positions)
    ax.set_xticklabels([name.replace(" ", "\n") for name in class_names], fontsize=8)
    ax.set_ylabel("Số ảnh")
    ax.set_title("Phân bố lớp theo tập dữ liệu", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> None:
    """Vẽ precision, recall và F1 thực tế cho từng lớp trên test."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    positions = np.arange(len(class_names))
    height = 0.24
    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, metric, color in (
        (-height, "precision", "#4C72B0"),
        (0, "recall", "#55A868"),
        (height, "f1-score", "#DD8452"),
    ):
        values = [float(report[name][metric]) for name in class_names]
        ax.barh(positions + offset, values, height, label=metric.title(), color=color)

    ax.set_yticks(positions)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Điểm số")
    ax.set_title("Precision, Recall và F1 theo lớp — tập Test", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_augmentation_examples(
    samples: list[tuple[np.ndarray, np.ndarray, str]],
    save_path: Path,
) -> bool:
    """Đặt cạnh nhau một số ảnh train thật và phiên bản tăng cường."""
    if not samples:
        return False

    fig, axes = plt.subplots(len(samples), 2, figsize=(7, 2.5 * len(samples)), squeeze=False)
    for row, (original, augmented, label) in enumerate(samples):
        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"Ảnh train · {label}", fontsize=9)
        axes[row, 1].imshow(augmented)
        axes[row, 1].set_title("Ảnh tăng cường", fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    fig.suptitle("Ví dụ tăng cường dữ liệu train", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_grid_search_results(
    all_results: dict[str, dict],
    save_dir: Path,
) -> bool:
    """Lưu điểm CV, validation trước/sau tinh chỉnh và siêu tham số tốt nhất."""
    rows = []
    for name, tuned in all_results.items():
        if not name.endswith("_tuned") or "cv_macro_f1" not in tuned:
            continue
        base_name = name.removesuffix("_tuned")
        baseline = all_results.get(base_name, {})
        rows.append({
            "model": base_name,
            "cv_macro_f1": float(tuned["cv_macro_f1"]),
            "validation_macro_f1_before": float(baseline.get("macro_f1", 0)),
            "validation_macro_f1_after": float(tuned.get("macro_f1", 0)),
            "best_params": tuned.get("best_params", {}),
        })

    if not rows:
        return False

    with (save_dir / "grid_search_results.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "model",
                "cv_macro_f1",
                "validation_macro_f1_before",
                "validation_macro_f1_after",
                "best_params",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "best_params": json.dumps(row["best_params"], ensure_ascii=False, sort_keys=True),
            })

    positions = np.arange(len(rows))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 2.5), 5))
    for index, (key, label, color) in enumerate((
        ("cv_macro_f1", "CV macro F1 tốt nhất", "#4C72B0"),
        ("validation_macro_f1_before", "Validation trước tinh chỉnh", "#55A868"),
        ("validation_macro_f1_after", "Validation sau tinh chỉnh", "#DD8452"),
    )):
        ax.bar(
            positions + (index - 1) * width,
            [row[key] for row in rows],
            width,
            label=label,
            color=color,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([row["model"] for row in rows], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title("Kiểm định chéo và tinh chỉnh siêu tham số", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "grid_search_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_model_timing(all_results: dict[str, dict], save_path: Path) -> bool:
    """So sánh thời gian fit và dự đoán/ảnh của các estimator ban đầu."""
    rows = [
        (name, values)
        for name, values in all_results.items()
        if not name.endswith("_tuned")
        and "train_time_s" in values
        and "infer_time_ms" in values
    ]
    if not rows:
        return False

    names = [name for name, _ in rows]
    positions = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(rows) * 0.42)), sharey=True)
    axes[0].barh(positions, [float(values["train_time_s"]) for _, values in rows], color="#4C72B0")
    axes[0].set_xlabel("Giây")
    axes[0].set_title("Thời gian fit sau tiền xử lý")
    axes[1].barh(positions, [float(values["infer_time_ms"]) for _, values in rows], color="#55A868")
    axes[1].set_xlabel("ms / ảnh")
    axes[1].set_title("Thời gian dự đoán")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Thời gian huấn luyện và suy luận theo mô hình", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_per_fruit_comparison(results: list[dict], save_path: Path) -> bool:
    """So sánh accuracy và Macro F1 test giữa model toàn cục/theo loại quả."""
    rows = [
        row for row in results
        if row.get("test_accuracy") not in (None, "")
        and row.get("test_macro_f1") not in (None, "")
    ]
    if not rows:
        return False

    labels = ["Toàn cục" if row["fruit"] == "all" else row["fruit"] for row in rows]
    positions = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 2.2), 5))
    accuracy = ax.bar(
        positions - width / 2,
        [float(row["test_accuracy"]) for row in rows],
        width,
        label="Accuracy",
        color="#4C72B0",
    )
    macro_f1 = ax.bar(
        positions + width / 2,
        [float(row["test_macro_f1"]) for row in rows],
        width,
        label="Macro F1",
        color="#DD8452",
    )
    ax.bar_label(accuracy, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(macro_f1, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Điểm số")
    ax.set_title("So sánh mô hình toàn cục và theo loại quả — Test", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


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
    if n_classes == 2:
        y_bin = np.column_stack(
            [np.asarray(y_true) == class_id for class_id in range(n_classes)]
        ).astype(np.int8)
    else:
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


def epoch_report(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: tuple[str, ...],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_images: list[np.ndarray],
    val_metrics: dict,
    save_dir: Path,
) -> dict[str, Any]:
    """Xuất biểu đồ và metrics kiểm thử cho model huấn luyện theo epoch."""
    save_dir.mkdir(parents=True, exist_ok=True)
    class_list = list(class_names)
    y_true = np.asarray(y_test)
    y_pred = np.asarray(model.predict(X_test))  # type: ignore[union-attr]
    if len(test_images) != len(y_true):
        raise ValueError("Số ảnh test phải bằng số nhãn test để lưu ảnh dự đoán sai.")

    _plot_dataset_distribution(
        np.asarray(train_labels),
        np.asarray(val_labels),
        y_true,
        class_list,
        save_dir / "data_distribution.png",
    )
    _plot_confusion_matrix(
        y_true, y_pred, class_list, save_dir / "confusion_matrix.png"
    )
    _plot_class_metrics(
        y_true, y_pred, class_list, save_dir / "class_metrics.png"
    )
    _save_classification_metrics_csv(
        y_true, y_pred, class_list, save_dir / "classification_metrics.csv"
    )
    _save_misclassified_images(
        test_images, y_true, y_pred, class_list, save_dir
    )

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)  # type: ignore[union-attr]
        aucs = _plot_roc_curves(
            y_true, y_score, class_list, save_dir / "roc_curves.png"
        )
        if aucs:
            with (save_dir / "roc_auc.csv").open(
                "w", newline="", encoding="utf-8-sig"
            ) as output:
                writer = csv.writer(output)
                writer.writerow(("class", "auc_ovr"))
                writer.writerows(aucs.items())

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_list))),
        target_names=class_list,
        zero_division=0,
    )
    (save_dir / "classification_report_SGDClassifier.txt").write_text(
        f"Model: SGDClassifier\n\n{report_text}",
        encoding="utf-8",
    )
    test_metrics = evaluate_predictions(
        y_true.tolist(), y_pred.tolist(), class_names=class_list
    )
    pair_counts: dict[tuple[str, str], int] = {}
    for true_id, predicted_id in zip(y_true, y_pred):
        if true_id != predicted_id:
            pair = (class_list[int(true_id)], class_list[int(predicted_id)])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    _write_summary(
        save_dir / "summary.md",
        "SGDClassifier",
        val_metrics,
        test_metrics,
        None,
        [
            (true_name, predicted_name, count)
            for (true_name, predicted_name), count in sorted(
                pair_counts.items(), key=lambda item: item[1], reverse=True
            )[:5]
        ],
    )
    return test_metrics


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
    augmentation_samples: list[tuple[np.ndarray, np.ndarray, str]] | None = None,
) -> None:
    """Tạo toàn bộ biểu đồ và in classification report."""
    save_dir.mkdir(parents=True, exist_ok=True)
    class_list = list(class_names)
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    print("\n── Xuất biểu đồ báo cáo ──")

    base_model_count = sum(not name.endswith("_tuned") for name in all_val_results)
    tuned_model_count = sum(
        name.endswith("_tuned") and "cv_macro_f1" in metrics
        for name, metrics in all_val_results.items()
    )
    _plot_training_pipeline(
        save_dir / "training_pipeline.png",
        model_count=base_model_count,
        tuned_count=tuned_model_count,
    )
    print(f"  Đã lưu: {save_dir / 'training_pipeline.png'}")
    _plot_inference_pipeline(save_dir / "inference_pipeline.png")
    print(f"  Đã lưu: {save_dir / 'inference_pipeline.png'}")

    if y_train is not None and y_val is not None:
        _plot_dataset_distribution(
            np.asarray(y_train),
            np.asarray(y_val),
            y_t,
            class_list,
            save_dir / "data_distribution.png",
        )
        print(f"  Đã lưu: {save_dir / 'data_distribution.png'}")

    if augmentation_samples:
        augmentation_path = save_dir / "augmentation_examples.png"
        if _plot_augmentation_examples(augmentation_samples, augmentation_path):
            print(f"  Đã lưu: {augmentation_path}")

    # Confusion matrix
    _plot_confusion_matrix(y_t, y_p, class_list, save_dir / "confusion_matrix.png")
    _plot_class_metrics(
        y_t, y_p, class_list, save_dir / "class_metrics.png"
    )
    print(f"  Đã lưu: {save_dir / 'class_metrics.png'}")

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
    if _plot_grid_search_results(all_val_results, save_dir):
        print(f"  Đã lưu: {save_dir / 'grid_search_results.png'}.")
    if _plot_model_timing(all_val_results, save_dir / "model_timing.png"):
        print(f"  Đã lưu: {save_dir / 'model_timing.png'}")
    if ablation_results:
        _save_ablation_report(ablation_results, save_dir)
    if per_fruit_results:
        _save_per_fruit_report(per_fruit_results, save_dir)
        if _plot_per_fruit_comparison(
            per_fruit_results, save_dir / "per_fruit_comparison.png"
        ):
            print(f"  Đã lưu: {save_dir / 'per_fruit_comparison.png'}")

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
