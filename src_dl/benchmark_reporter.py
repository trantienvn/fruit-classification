"""
benchmark_reporter.py — Đo lường hiệu năng kỹ thuật và tự động xuất 6 báo cáo học thuật.

Tự động xuất 6 tệp:
    1. metrics.json: Đánh giá chi tiết (Accuracy, Macro F1, Precision, Recall, Per-class).
    2. confusion_matrix.png: Ma trận nhầm lẫn 9x9 trực quan.
    3. roc_curve.png: Đường cong ROC đa lớp (One-vs-Rest) và chỉ số AUC.
    4. training_history.png: Diễn tiến Loss và Accuracy (Train vs Val) theo Epoch.
    5. model_comparison.png: Biểu đồ so sánh đa chiều giữa các mô hình học sâu.
    6. benchmark.csv: Tốc độ suy luận, kích thước, FLOPs, số tham số và RAM/VRAM.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")  # Backend không cần GUI
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Nhãn tiếng Việt chuẩn 9 lớp
CLASS_NAMES_VI = [
    "Cam chín",
    "Cam xanh",
    "Cam hỏng",
    "Chuối chín",
    "Chuối xanh",
    "Chuối hỏng",
    "Xoài chín",
    "Xoài xanh",
    "Xoài hỏng",
]


# ===========================================================================
# 1. ĐO LƯỜNG FLOPs VÀ THAM SỐ (Pure PyTorch, không cần thư viện ngoài)
# ===========================================================================

def count_model_flops_and_params(
    model: Any,
    input_size: tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
) -> tuple[float, float]:
    """
    Tính số triệu tham số (Parameters in M) và FLOPs (MFLOPs) của mô hình.
    Sử dụng forward hook trên Conv2d, Linear, BatchNorm2d, Pooling.

    Trả về:
        (params_m, flops_mflops)
    """
    import torch
    import torch.nn as nn

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    params_m = round(total_params / 1e6, 3)

    total_flops = [0]
    hooks = []

    def conv_hook(m: nn.Conv2d, inp: Any, out: torch.Tensor) -> None:
        batch_size, out_c, out_h, out_w = out.shape
        in_c = m.in_channels
        k_h, k_w = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        groups = m.groups
        flops = 2 * (in_c // groups) * k_h * k_w * out_c * out_h * out_w
        if m.bias is not None:
            flops += out_c * out_h * out_w
        total_flops[0] += flops * batch_size

    def linear_hook(m: nn.Linear, inp: Any, out: torch.Tensor) -> None:
        batch_size = out.shape[0] if out.ndim > 1 else 1
        flops = 2 * m.in_features * m.out_features
        if m.bias is not None:
            flops += m.out_features
        total_flops[0] += flops * batch_size

    def bn_hook(m: Any, inp: Any, out: torch.Tensor) -> None:
        total_flops[0] += 2 * out.numel()

    def pool_hook(m: Any, inp: Any, out: torch.Tensor) -> None:
        total_flops[0] += out.numel()

    # Gắn hook vào các tầng con
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(linear_hook))
        elif isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            hooks.append(mod.register_forward_hook(bn_hook))
        elif isinstance(mod, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
            hooks.append(mod.register_forward_hook(pool_hook))

    # Chạy dummy input 1 lần
    dummy = torch.zeros(input_size, dtype=torch.float32, device=dev)
    try:
        with torch.no_grad():
            _ = model(dummy)
    finally:
        for h in hooks:
            h.remove()

    flops_mflops = round(total_flops[0] / 1e6, 2)
    return params_m, flops_mflops


# ===========================================================================
# 2. ĐO LƯỜNG ĐỘ TRỄ SUY LUẬN (LATENCY) VÀ RAM/VRAM
# ===========================================================================

def measure_latency_and_ram(
    model: Any,
    input_size: tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cuda",
    warmup: int = 10,
    reps: int = 50,
) -> tuple[float, float]:
    """
    Đo thời gian suy luận trung bình (Latency - ms) và Peak Memory (MB).

    Trả về:
        (avg_latency_ms, peak_mem_mb)
    """
    import torch

    target_dev = torch.device(device)
    if target_dev.type == "cuda" and not torch.cuda.is_available():
        target_dev = torch.device("cpu")

    model = model.to(target_dev)
    model.eval()
    dummy = torch.randn(input_size, dtype=torch.float32, device=target_dev)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    # Đo bộ nhớ
    peak_mem_mb = 0.0
    if target_dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_dev)
        with torch.no_grad():
            _ = model(dummy)
        peak_mem_mb = round(torch.cuda.max_memory_allocated(target_dev) / (1024 * 1024), 2)
    else:
        try:
            import psutil
            process = psutil.Process(os.getpid())
            peak_mem_mb = round(process.memory_info().rss / (1024 * 1024), 2)
        except Exception:
            peak_mem_mb = 0.0

    # Đo độ trễ
    latencies: list[float] = []
    if target_dev.type == "cuda":
        torch.cuda.synchronize(target_dev)
        for _ in range(reps):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy)
            torch.cuda.synchronize(target_dev)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
    else:
        for _ in range(reps):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    avg_latency_ms = round(float(np.mean(latencies)), 2)
    return avg_latency_ms, peak_mem_mb


# ===========================================================================
# 3. XUẤT METRICS.JSON (Mục 1)
# ===========================================================================

def export_metrics_json(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
    model_name: str = "TorchMLP",
    extra_info: dict | None = None,
) -> dict:
    """Tính toán và xuất đầy đủ chỉ số đánh giá ra tệp JSON."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    per_class_dict = {}
    for idx, name in enumerate(class_names):
        vi_name = CLASS_NAMES_VI[idx] if idx < len(CLASS_NAMES_VI) else name
        per_class_dict[name] = {
            "name_vi": vi_name,
            "precision": round(float(p_per[idx]), 4),
            "recall": round(float(r_per[idx]), 4),
            "f1_score": round(float(f1_per[idx]), 4),
            "support": int(sup_per[idx]),
        }

    metrics_data = {
        "model_name": model_name,
        "overall": {
            "accuracy": round(acc, 4),
            "accuracy_percent": f"{acc * 100:.2f}%",
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "total_test_samples": int(len(y_true)),
        },
        "per_class": per_class_dict,
    }

    if extra_info:
        metrics_data["extra_info"] = extra_info

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)

    logger.info(f"[benchmark_reporter] Đã xuất metrics: {save_path}")
    return metrics_data


# ===========================================================================
# 4. VẼ VÀ XUẤT MA TRẬN NHẦM LẪN CONFUSION_MATRIX.PNG (Mục 2)
# ===========================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str = "Ma trận nhầm lẫn (Confusion Matrix)",
) -> None:
    """Vẽ và xuất ma trận nhầm lẫn kích thước 9x9 với nhãn tiếng Việt."""
    from sklearn.metrics import confusion_matrix

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    display_labels = [
        CLASS_NAMES_VI[i] if i < len(CLASS_NAMES_VI) else class_names[i]
        for i in range(len(class_names))
    ]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Tỷ lệ chuẩn hóa (%)", rotation=270, labelpad=15, fontsize=10)

    # Đặt trục
    tick_marks = np.arange(len(display_labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(display_labels, rotation=35, ha="right", fontsize=9, fontweight="medium")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(display_labels, fontsize=9, fontweight="medium")

    # Thêm số lượng + phần trăm vào từng ô
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(
                j, i,
                f"{val}\n({pct:.1f}%)" if val > 0 else "0",
                ha="center", va="center",
                color=color, fontsize=8,
            )

    ax.set_ylabel("Nhãn thực tế (True Label)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_xlabel("Nhãn dự đoán (Predicted Label)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[benchmark_reporter] Đã xuất ma trận nhầm lẫn: {save_path}")


# ===========================================================================
# 5. VẼ ĐƯỜNG CONG ROC_CURVE.PNG (Mục 3)
# ===========================================================================

def plot_roc_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str = "Đường cong ROC đa lớp (One-vs-Rest) & Chỉ số AUC",
) -> None:
    """Vẽ đường cong ROC cho cả 9 lớp + Macro-average AUC."""
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr: dict[int, np.ndarray] = {}
    tpr: dict[int, np.ndarray] = {}
    roc_auc: dict[int, float] = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    # Đường chéo ngẫu nhiên
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Dự đoán ngẫu nhiên (AUC = 0.50)")

    # Macro-average
    ax.plot(
        all_fpr, mean_tpr,
        label=f"Macro-average (AUC = {macro_auc:.4f})",
        color="navy", linestyle=":", linewidth=2.5,
    )

    # Từng lớp
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i in range(n_classes):
        vi_name = CLASS_NAMES_VI[i] if i < len(CLASS_NAMES_VI) else class_names[i]
        ax.plot(
            fpr[i], tpr[i],
            color=colors[i], lw=1.5,
            label=f"{vi_name} (AUC = {roc_auc[i]:.4f})",
        )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Tỷ lệ dương tính giả (False Positive Rate - FPR)", fontsize=11, labelpad=8)
    ax.set_ylabel("Tỷ lệ dương tính thật (True Positive Rate - TPR)", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[benchmark_reporter] Đã xuất ROC curves: {save_path}")


# ===========================================================================
# 6. VẼ LỊCH SỬ HUẤN LUYỆN TRAINING_HISTORY.PNG (Mục 4)
# ===========================================================================

def plot_training_history(
    history: Sequence[dict[str, Any]],
    save_path: Path,
    title: str = "Lịch sử huấn luyện (Training & Validation Curves)",
) -> None:
    """
    Vẽ 2 đồ thị con:
        Subplot 1: Train Loss vs Val Loss
        Subplot 2: Train Accuracy vs Val Accuracy
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        logger.warning("[benchmark_reporter] Lịch sử huấn luyện rỗng, bỏ qua vẽ đồ thị.")
        return

    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
    train_loss = [h.get("train_loss", 0.0) for h in history]
    val_loss   = [h.get("val_loss", 0.0) for h in history]
    train_acc  = [h.get("train_acc", 0.0) for h in history]
    val_acc    = [h.get("val_acc", 0.0) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    # Subplot 1: Loss
    ax1.plot(epochs, train_loss, "b-o", markersize=4, label="Train Loss", lw=1.8)
    ax1.plot(epochs, val_loss, "r-s", markersize=4, label="Validation Loss", lw=1.8)
    best_loss_idx = int(np.argmin(val_loss))
    ax1.scatter(
        epochs[best_loss_idx], val_loss[best_loss_idx],
        color="crimson", s=100, zorder=5,
        label=f"Min Val Loss: {val_loss[best_loss_idx]:.4f} (Ep {epochs[best_loss_idx]})",
    )
    ax1.set_title("Hàm mất mát (Cross-Entropy Loss)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Accuracy
    ax2.plot(epochs, train_acc, "b-o", markersize=4, label="Train Accuracy", lw=1.8)
    ax2.plot(epochs, val_acc, "g-^", markersize=4, label="Validation Accuracy", lw=1.8)
    best_acc_idx = int(np.argmax(val_acc))
    ax2.scatter(
        epochs[best_acc_idx], val_acc[best_acc_idx],
        color="darkgreen", s=100, zorder=5,
        label=f"Max Val Acc: {val_acc[best_acc_idx]:.2f}% (Ep {epochs[best_acc_idx]})",
    )
    ax2.set_title("Độ chính xác (Accuracy %)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[benchmark_reporter] Đã xuất lịch sử huấn luyện: {save_path}")


# ===========================================================================
# 7. VẼ SO SÁNH CÁC MÔ HÌNH MODEL_COMPARISON.PNG (Mục 5)
# ===========================================================================

def plot_model_comparison(
    models_data: list[dict[str, Any]],
    save_path: Path,
    title: str = "So sánh hiệu năng giữa các kiến trúc học sâu (Mục 3.6 Đề cương)",
) -> None:
    """
    Vẽ biểu đồ tổng hợp 4 chỉ số chính:
        1. Độ chính xác & F1-score (%)
        2. Kích thước mô hình (MB)
        3. Tốc độ suy luận (ms)
        4. Số lượng tham số (M) & FLOPs (MFLOPs)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not models_data:
        logger.warning("[benchmark_reporter] Dữ liệu so sánh rỗng, bỏ qua.")
        return

    names = [m["name"] for m in models_data]
    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)

    # 1. Accuracy vs F1
    accs = [m.get("accuracy", 0.0) * 100 for m in models_data]
    f1s  = [m.get("macro_f1", 0.0) * 100 for m in models_data]
    ax1 = axes[0, 0]
    b1 = ax1.bar(x - width/2, accs, width, label="Accuracy (%)", color="#2563eb")
    b2 = ax1.bar(x + width/2, f1s, width, label="Macro F1 (%)", color="#10b981")
    ax1.set_ylabel("Tỷ lệ (%)", fontsize=10)
    ax1.set_title("1. Độ chính xác & Macro F1 trên tập Test", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10, fontweight="medium")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_ylim([max(0, min(accs + f1s) - 10), 105])
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # 2. Dung lượng mô hình (MB)
    sizes = [m.get("size_mb", 0.0) for m in models_data]
    ax2 = axes[0, 1]
    b3 = ax2.bar(x, sizes, width * 1.2, color="#8b5cf6")
    ax2.set_ylabel("Dung lượng (MB)", fontsize=10)
    ax2.set_title("2. Kích thước mô hình trên đĩa", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=10, fontweight="medium")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    for b in b3:
        h = b.get_height()
        ax2.annotate(f"{h:.1f} MB", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # 3. Tốc độ suy luận (Latency ms)
    lat_gpu = [m.get("latency_gpu_ms", 0.0) for m in models_data]
    lat_cpu = [m.get("latency_cpu_ms", 0.0) for m in models_data]
    ax3 = axes[1, 0]
    b4 = ax3.bar(x - width/2, lat_gpu, width, label="GPU RTX 3050 (ms)", color="#f59e0b")
    b5 = ax3.bar(x + width/2, lat_cpu, width, label="CPU (ms)", color="#06b6d4")
    ax3.set_ylabel("Độ trễ suy luận (ms)", fontsize=10)
    ax3.set_title("3. Tốc độ suy luận trung bình 1 ảnh", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, fontsize=10, fontweight="medium")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.5)
    for b in list(b4) + list(b5):
        h = b.get_height()
        if h > 0:
            ax3.annotate(f"{h:.1f}ms", xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # 4. FLOPs & Tham số
    flops = [m.get("flops_mflops", 0.0) for m in models_data]
    ax4 = axes[1, 1]
    b6 = ax4.bar(x, flops, width * 1.2, color="#ec4899")
    ax4.set_ylabel("Độ phức tạp (MFLOPs)", fontsize=10)
    ax4.set_title("4. Số phép tính dấu phẩy động (FLOPs)", fontsize=11, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, fontsize=10, fontweight="medium")
    ax4.grid(True, axis="y", linestyle="--", alpha=0.5)
    for b in b6:
        h = b.get_height()
        ax4.annotate(f"{h:.0f}M", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[benchmark_reporter] Đã xuất so sánh mô hình: {save_path}")


# ===========================================================================
# 8. XUẤT BENCHMARK.CSV (Mục 6)
# ===========================================================================

def export_benchmark_csv(
    models_data: list[dict[str, Any]],
    save_path: Path,
) -> None:
    """Xuất bảng tổng hợp hiệu năng đo lường kỹ thuật ra tệp CSV."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Model",
        "Parameters (M)",
        "FLOPs (MFLOPs)",
        "Model Size (MB)",
        "Inference GPU (ms)",
        "Inference CPU (ms)",
        "Peak RAM/VRAM (MB)",
        "Test Accuracy (%)",
        "Test Macro F1 (%)",
    ]

    rows = []
    for m in models_data:
        rows.append({
            "Model": m.get("name", "Unknown"),
            "Parameters (M)": m.get("params_m", "N/A"),
            "FLOPs (MFLOPs)": m.get("flops_mflops", "N/A"),
            "Model Size (MB)": m.get("size_mb", "N/A"),
            "Inference GPU (ms)": m.get("latency_gpu_ms", "N/A"),
            "Inference CPU (ms)": m.get("latency_cpu_ms", "N/A"),
            "Peak RAM/VRAM (MB)": m.get("peak_mem_mb", "N/A"),
            "Test Accuracy (%)": f"{m.get('accuracy', 0.0) * 100:.2f}%",
            "Test Macro F1 (%)": f"{m.get('macro_f1', 0.0) * 100:.2f}%",
        })

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"[benchmark_reporter] Đã xuất benchmark CSV: {save_path}")
