"""
test_benchmark_reporter.py — Kiểm tra bộ công cụ đo lường và xuất 6 tệp báo cáo học thuật.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src_dl.backbone import get_backbone, SUPPORTED_BACKBONES
from src_dl.benchmark_reporter import (
    CLASS_NAMES_VI,
    count_model_flops_and_params,
    export_benchmark_csv,
    export_metrics_json,
    measure_latency_and_ram,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curves,
    plot_training_history,
)


def test_flops_and_params_calculation():
    """Kiểm tra tính toán FLOPs và tham số cho 3 backbone."""
    for name in SUPPORTED_BACKBONES:
        bb = get_backbone(name, device="cpu")
        extractor = bb.get_torch_feature_extractor()
        params_m, flops_mflops = count_model_flops_and_params(extractor, (1, 3, 224, 224), device="cpu")
        assert params_m > 0.5, f"{name}: params_m quá nhỏ ({params_m})"
        assert flops_mflops > 10.0, f"{name}: flops_mflops quá nhỏ ({flops_mflops})"
        print(f"[{name}] Params: {params_m}M | FLOPs: {flops_mflops} MFLOPs")


def test_reports_generation(tmp_path: Path):
    """Kiểm tra xuất đầy đủ 6 tệp báo cáo vào thư mục tạm."""
    # Giả lập 270 mẫu test (9 lớp)
    np.random.seed(42)
    n_samples = 270
    n_classes = 9
    y_true = np.random.randint(0, n_classes, size=n_samples)
    
    # Tạo xác suất softmax giả lập có độ chính xác cao
    y_scores = np.zeros((n_samples, n_classes), dtype=np.float32)
    for i in range(n_samples):
        # 85% xác suất gán đúng lớp
        correct_cls = y_true[i] if np.random.rand() < 0.85 else np.random.randint(0, n_classes)
        y_scores[i, correct_cls] += 3.0
        y_scores[i] += np.random.exponential(0.5, size=n_classes)
    y_scores /= y_scores.sum(axis=1, keepdims=True)
    y_pred = np.argmax(y_scores, axis=1)

    class_names = [f"class_{i}" for i in range(n_classes)]

    # 1. metrics.json
    metrics_path = tmp_path / "metrics.json"
    metrics = export_metrics_json(y_true, y_pred, class_names, metrics_path, model_name="TestModel")
    assert metrics_path.exists()
    assert "accuracy" in metrics["overall"]
    assert len(metrics["per_class"]) == 9

    # 2. confusion_matrix.png
    cm_path = tmp_path / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    assert cm_path.exists()
    assert cm_path.stat().st_size > 1000

    # 3. roc_curve.png
    roc_path = tmp_path / "roc_curve.png"
    plot_roc_curves(y_true, y_scores, class_names, roc_path)
    assert roc_path.exists()
    assert roc_path.stat().st_size > 1000

    # 4. training_history.png
    history = [
        {"epoch": i + 1, "train_loss": 1.0 / (i + 1), "val_loss": 1.2 / (i + 1),
         "train_acc": 50 + i * 3, "val_acc": 48 + i * 3}
        for i in range(15)
    ]
    history_path = tmp_path / "training_history.png"
    plot_training_history(history, history_path)
    assert history_path.exists()
    assert history_path.stat().st_size > 1000

    # 5. model_comparison.png & 6. benchmark.csv
    models_data = [
        {
            "name": "MobileNetV3-Small",
            "params_m": 1.5,
            "flops_mflops": 58.0,
            "size_mb": 4.26,
            "latency_gpu_ms": 3.2,
            "latency_cpu_ms": 18.5,
            "peak_mem_mb": 45.0,
            "accuracy": 0.963,
            "macro_f1": 0.961,
        },
        {
            "name": "MobileNetV2",
            "params_m": 2.2,
            "flops_mflops": 300.0,
            "size_mb": 11.4,
            "latency_gpu_ms": 5.1,
            "latency_cpu_ms": 28.2,
            "peak_mem_mb": 62.0,
            "accuracy": 0.974,
            "macro_f1": 0.972,
        },
        {
            "name": "ResNet-18",
            "params_m": 11.2,
            "flops_mflops": 1814.0,
            "size_mb": 44.7,
            "latency_gpu_ms": 6.8,
            "latency_cpu_ms": 42.0,
            "peak_mem_mb": 95.0,
            "accuracy": 0.981,
            "macro_f1": 0.980,
        },
    ]

    comp_path = tmp_path / "model_comparison.png"
    plot_model_comparison(models_data, comp_path)
    assert comp_path.exists()
    assert comp_path.stat().st_size > 1000

    bench_path = tmp_path / "benchmark.csv"
    export_benchmark_csv(models_data, bench_path)
    assert bench_path.exists()
    assert bench_path.stat().st_size > 50

    print("Tất cả 6 tệp kiểm tra sinh thành công tại:", tmp_path)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_flops_and_params_calculation()
        test_reports_generation(Path(tmp_dir))
    print("✓ Toàn bộ test của Giai đoạn 1 đã vượt qua thành công!")
