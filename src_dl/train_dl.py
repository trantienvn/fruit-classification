"""
train_dl.py — Huấn luyện và so sánh các mô hình Deep Learning (MobileNetV3, MobileNetV2, ResNet-18).

Tính năng:
    - Hỗ trợ huấn luyện so sánh 3 kiến trúc học sâu: MobileNetV3-Small, MobileNetV2, ResNet-18 (--backbone all).
    - Huấn luyện mạng nơ-ron sâu TorchMLPClassifier trên NVIDIA CUDA GPU (RTX 3050).
    - Tự động xuất đầy đủ 6 tệp báo cáo học thuật theo đề cương (Mục 1.5 & Chương 3):
        1. reports_dl/metrics.json: Đánh giá chi tiết (Accuracy, Macro F1, Precision, Recall).
        2. reports_dl/confusion_matrix.png: Ma trận nhầm lẫn 9x9 trực quan.
        3. reports_dl/roc_curve.png: Đường cong ROC đa lớp (One-vs-Rest) và chỉ số AUC.
        4. reports_dl/training_history.png: Diễn tiến Loss và Accuracy qua từng Epoch.
        5. reports_dl/model_comparison.png: Biểu đồ so sánh đa chiều giữa các mô hình.
        6. reports_dl/benchmark.csv: Tốc độ suy luận, kích thước, FLOPs, số tham số và RAM/VRAM.
    - Lưu checkpoint mô hình tốt nhất vào models_dl/classifier_dl.joblib và models_dl/config.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src_dl.backbone import (
    BACKBONE_NAME,
    BACKBONE_SPECS,
    DEFAULT_BACKBONE,
    EMBEDDING_DIM,
    SUPPORTED_BACKBONES,
    get_backbone,
)
from src_dl.benchmark_reporter import (
    count_model_flops_and_params,
    export_benchmark_csv,
    export_metrics_json,
    measure_latency_and_ram,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curves,
    plot_training_history,
)
from src_dl.preprocess_dl import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    cache_is_valid,
    extract_and_cache,
    get_cache_dir_for_backbone,
    load_cached_features,
)
from src_dl.torch_classifier import TorchMLPClassifier, plot_loss_curve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số mặc định
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR:  Path = Path("models_dl")
DEFAULT_REPORT_DIR: Path = Path("reports_dl")

PCA_RATIO: float = 0.95      # giữ 95% phương sai


def _build_pipeline(estimator: Any, use_pca: bool = True) -> Pipeline:
    """Tạo sklearn Pipeline gồm StandardScaler, PCA (tùy chọn) và Estimator."""
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=PCA_RATIO, random_state=42)))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def _sync_report_to_web(report_dir: Path) -> None:
    """Đồng bộ toàn bộ báo cáo và biểu đồ sang thư mục reports/ và frontend/reports/."""
    targets = [Path("reports"), Path("frontend/reports")]
    for target_dir in targets:
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in report_dir.glob("*"):
            if f.is_file():
                try:
                    shutil.copy2(f, target_dir / f.name)
                except Exception as exc:
                    logger.debug(f"Không thể sao chép {f.name} sang {target_dir}: {exc}")


def _benchmark_pipeline(
    backbone_name: str,
    pipeline: Pipeline,
    model_path: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Đo lường FLOPs, Parameters, Latency (GPU & CPU), RAM/VRAM và Size của mô hình."""
    import torch

    spec = BACKBONE_SPECS.get(backbone_name, {})
    display_name = spec.get("display_name", backbone_name)

    # 1. FLOPs & Params
    bb = get_backbone(backbone_name, device="cpu")
    bb_ext = bb.get_torch_feature_extractor()
    bb_params, bb_flops = count_model_flops_and_params(bb_ext, (1, 3, 224, 224), device="cpu")

    clf: TorchMLPClassifier = pipeline.named_steps["clf"]
    mlp_params, mlp_flops = 0.0, 0.0
    if hasattr(clf, "_net") and clf._net is not None:
        first_layer = getattr(clf._net, "fc1", None) or clf._net[0] if isinstance(clf._net, torch.nn.Sequential) else None
        in_dim = getattr(first_layer, "in_features", 179)
        mlp_net_cpu = clf._net.to(torch.device("cpu"))
        mlp_params, mlp_flops = count_model_flops_and_params(mlp_net_cpu, (1, in_dim), device="cpu")

    total_params = round(bb_params + mlp_params, 3)
    total_flops  = round(bb_flops + mlp_flops, 2)

    # 2. Latency & Memory
    gpu_lat, peak_mem = 0.0, 0.0
    if device in ("cuda", "auto") and torch.cuda.is_available():
        gpu_lat, peak_mem = measure_latency_and_ram(bb_ext, (1, 3, 224, 224), device="cuda", warmup=5, reps=30)
    cpu_lat, _ = measure_latency_and_ram(bb_ext, (1, 3, 224, 224), device="cpu", warmup=3, reps=15)

    # 3. Model Size
    size_mb = round(model_path.stat().st_size / (1024 * 1024), 2) if model_path.exists() else 0.0

    return {
        "name": display_name,
        "backbone_name": backbone_name,
        "params_m": total_params,
        "flops_mflops": total_flops,
        "size_mb": size_mb,
        "latency_gpu_ms": gpu_lat,
        "latency_cpu_ms": cpu_lat,
        "peak_mem_mb": peak_mem,
    }


def train_single_backbone(
    backbone_name: str,
    data_dir: Path,
    model_dir: Path,
    report_dir: Path,
    epochs: int = 15,
    lr: float = 1e-3,
    device: str = "auto",
    use_pca: bool = True,
    force_preprocess: bool = False,
) -> dict[str, Any]:
    """Huấn luyện và đánh giá 1 mô hình với backbone chỉ định."""
    import torch

    cache_dir = get_cache_dir_for_backbone(backbone_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    spec = BACKBONE_SPECS[backbone_name]
    display_name = spec["display_name"]
    logger.info(f"\n{'='*70}\n[train_dl] Bắt đầu xử lý {display_name} ({backbone_name})\n{'='*70}")

    # Bước 1: Trích xuất / kiểm tra cache
    if force_preprocess or not cache_is_valid(cache_dir, data_dir, backbone_name):
        logger.info(f"[train_dl] Trích xuất đặc trưng cho {backbone_name}...")
        extract_and_cache(
            data_dir=data_dir,
            cache_dir=cache_dir,
            backbone_name=backbone_name,
            force=force_preprocess,
            device=device,
        )

    # Bước 2: Tải dữ liệu
    data = load_cached_features(cache_dir)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val     = data["X_val"], data["y_val"]
    X_test, y_test   = data["X_test"], data["y_test"]
    class_names      = data["class_names"]

    # Bước 3: Huấn luyện TorchMLP
    clf = TorchMLPClassifier(
        device=device,
        epochs=epochs,
        lr=lr,
        random_state=42,
    )
    pipeline = _build_pipeline(clf, use_pca=use_pca)

    # Tiền xử lý dữ liệu trước khi nạp vào TorchMLP
    X_train_proc = X_train
    X_val_proc   = X_val
    for step_name in ("scaler", "pca"):
        if step_name in pipeline.named_steps:
            transformer = pipeline.named_steps[step_name]
            X_train_proc = transformer.fit_transform(X_train_proc)
            if X_val.shape[0] > 0:
                X_val_proc = transformer.transform(X_val_proc)

    eval_set = (X_val_proc, y_val) if X_val.shape[0] > 0 else None
    logger.info(f"[train_dl] Huấn luyện TorchMLPClassifier ({epochs} epochs) trên {device}...")
    t0 = time.perf_counter()
    clf.fit(X_train_proc, y_train, eval_set=eval_set)
    train_time = round(time.perf_counter() - t0, 2)
    logger.info(f"[train_dl] {display_name} hoàn tất huấn luyện trong {train_time}s")

    # Bước 4: Đánh giá trên Validation & Test
    y_pred_val = pipeline.predict(X_val) if X_val.shape[0] > 0 else y_train[:1]
    val_acc    = float(accuracy_score(y_val, y_pred_val)) if X_val.shape[0] > 0 else 0.0
    val_f1     = float(f1_score(y_val, y_pred_val, average="macro", zero_division=0)) if X_val.shape[0] > 0 else 0.0

    y_pred_test  = pipeline.predict(X_test)
    y_scores_test = pipeline.predict_proba(X_test)
    test_acc     = float(accuracy_score(y_test, y_pred_test))
    test_f1      = float(f1_score(y_test, y_pred_test, average="macro", zero_division=0))

    logger.info(f"[train_dl] {display_name} Kết quả Test: Accuracy={test_acc*100:.2f}% | Macro F1={test_f1*100:.2f}%")

    # Bước 5: Lưu checkpoint mô hình
    model_save_path = model_dir / f"classifier_{backbone_name}.joblib"
    joblib.dump(pipeline, model_save_path)
    logger.info(f"[train_dl] Đã lưu mô hình: {model_save_path}")

    # Bước 6: Đo lường Benchmark
    bench_data = _benchmark_pipeline(backbone_name, pipeline, model_save_path, device=device)
    bench_data.update({
        "train_time_s": train_time,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "accuracy": test_acc,
        "macro_f1": test_f1,
    })

    # Bước 7: Xuất báo cáo đồ thị riêng cho mô hình này
    history = getattr(clf, "history_", [])
    plot_training_history(
        history,
        report_dir / f"training_history_{backbone_name}.png",
        title=f"Lịch sử huấn luyện — {display_name} ({len(history)} Epochs)",
    )
    plot_confusion_matrix(
        y_test, y_pred_test, class_names,
        report_dir / f"confusion_matrix_{backbone_name}.png",
        title=f"Ma trận nhầm lẫn — {display_name} (Test Set)",
    )
    plot_roc_curves(
        y_test, y_scores_test, class_names,
        report_dir / f"roc_curve_{backbone_name}.png",
        title=f"Đường cong ROC — {display_name} (One-vs-Rest)",
    )
    export_metrics_json(
        y_test, y_pred_test, class_names,
        report_dir / f"metrics_{backbone_name}.json",
        model_name=display_name,
        extra_info=bench_data,
    )

    return {
        "backbone_name": backbone_name,
        "display_name": display_name,
        "pipeline": pipeline,
        "clf": clf,
        "model_path": model_save_path,
        "bench_data": bench_data,
        "history": history,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "y_scores_test": y_scores_test,
        "class_names": class_names,
        "val_f1": val_f1,
        "test_f1": test_f1,
        "test_acc": test_acc,
    }


def train(
    model_dir:        Path = DEFAULT_MODEL_DIR,
    report_dir:       Path = DEFAULT_REPORT_DIR,
    cache_dir:        Path = DEFAULT_CACHE_DIR,
    data_dir:         Path = DEFAULT_DATA_DIR,
    backbone:         str  = "all",
    use_pca:          bool = True,
    no_gridsearch:    bool = False,
    force_preprocess: bool = False,
    top_k:            int  = 3,
    device:           str  = "auto",
    model_mode:       str  = "torch",
    epochs:           int  = 15,
    lr:               float | None = None,
    resume:           bool = False,
) -> dict[str, Any]:
    """
    Quy trình huấn luyện chính:
        - Nếu backbone='all': Huấn luyện tuần tự cả 3 mô hình (MobileNetV3, MobileNetV2, ResNet18)
        - Đo lường và sinh đầy đủ 6 tệp: metrics.json, confusion_matrix.png, roc_curve.png,
          training_history.png, model_comparison.png, benchmark.csv.
        - Chọn mô hình tốt nhất lưu làm classifier_dl.joblib mặc định cho web.
    """
    import torch

    model_dir  = Path(model_dir)
    report_dir = Path(report_dir)
    data_dir   = Path(data_dir)

    effective_lr = lr if lr is not None else 1e-3

    # Xác định danh sách backbones cần chạy
    norm_backbone = backbone.lower().strip()
    if norm_backbone == "all":
        target_backbones = list(SUPPORTED_BACKBONES)
    else:
        if norm_backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Backbone '{backbone}' không hợp lệ. Chọn một trong: {['all'] + SUPPORTED_BACKBONES}")
        target_backbones = [norm_backbone]

    logger.info(
        f"\n[train_dl] KHỞI CHẠY HUẤN LUYỆN ĐA KIẾN TRÚC\n"
        f"  Danh sách mô hình: {[BACKBONE_SPECS[b]['display_name'] for b in target_backbones]}\n"
        f"  Epochs           : {epochs}\n"
        f"  Learning Rate    : {effective_lr}\n"
        f"  Device           : {device}\n"
        f"  PCA              : {use_pca} ({PCA_RATIO*100:.0f}% variance)\n"
    )

    all_trained: list[dict[str, Any]] = []
    models_comparison_data: list[dict[str, Any]] = []

    for b_name in target_backbones:
        res = train_single_backbone(
            backbone_name=b_name,
            data_dir=data_dir,
            model_dir=model_dir,
            report_dir=report_dir,
            epochs=epochs,
            lr=effective_lr,
            device=device,
            use_pca=use_pca,
            force_preprocess=force_preprocess,
        )
        all_trained.append(res)
        models_comparison_data.append(res["bench_data"])

    # Chọn mô hình tốt nhất (dựa trên Macro F1 trên tập Test)
    all_trained.sort(key=lambda x: x["test_f1"], reverse=True)
    best = all_trained[0]
    logger.info(
        f"\n{'='*70}\n[train_dl] MÔ HÌNH XUẤT SẮC NHẤT: {best['display_name']} "
        f"(Test F1 = {best['test_f1']*100:.2f}%, Test Acc = {best['test_acc']*100:.2f}%)\n{'='*70}"
    )

    # Lưu mô hình tốt nhất làm classifier_dl.joblib
    default_joblib = model_dir / "classifier_dl.joblib"
    shutil.copy2(best["model_path"], default_joblib)
    logger.info(f"[train_dl] Đã lưu mô hình tốt nhất: {default_joblib}")

    # Lưu config.json
    best_spec = BACKBONE_SPECS[best["backbone_name"]]
    config = {
        "feature_type": f"cnn_{best['backbone_name']}",
        "backbone": best["backbone_name"],
        "display_name": best["display_name"],
        "embedding_dim": best_spec["embedding_dim"],
        "crop_size": list(best_spec["input_size"]),
        "class_names": best["class_names"],
        "n_classes": len(best["class_names"]),
        "best_model": f"{best['display_name']} + TorchMLP",
        "best_val_f1": round(best["val_f1"], 6),
        "test_accuracy": round(best["test_acc"], 6),
        "test_f1": round(best["test_f1"], 6),
        "test_metrics": {
            "accuracy": round(best["test_acc"], 6),
            "macro_f1": round(best["test_f1"], 6),
        },
        "total_epochs": epochs,
        "use_pca": use_pca,
        "pca_ratio": PCA_RATIO,
        "all_models_compared": [m["name"] for m in models_comparison_data],
    }
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------------
    # TỰ ĐỘNG XUẤT 6 TỆP BÁO CÁO CHÍNH THỨC (THEO ĐÚNG YÊU CẦU ĐỀ CƯƠNG)
    # -----------------------------------------------------------------------
    logger.info("[train_dl] Đang tự động xuất 6 tệp báo cáo học thuật chuẩn...")

    # 1. metrics.json (của mô hình tốt nhất kèm tóm tắt so sánh)
    export_metrics_json(
        best["y_test"], best["y_pred_test"], best["class_names"],
        report_dir / "metrics.json",
        model_name=best["display_name"],
        extra_info={"comparison": models_comparison_data},
    )

    # 2. confusion_matrix.png (của mô hình tốt nhất)
    plot_confusion_matrix(
        best["y_test"], best["y_pred_test"], best["class_names"],
        report_dir / "confusion_matrix.png",
        title=f"Ma trận nhầm lẫn tổng thể — {best['display_name']} (Mục 1.5.1 Đề cương)",
    )

    # 3. roc_curve.png (của mô hình tốt nhất)
    plot_roc_curves(
        best["y_test"], best["y_scores_test"], best["class_names"],
        report_dir / "roc_curve.png",
        title=f"Đường cong ROC & AUC đa lớp — {best['display_name']} (Mục 1.5.2 Đề cương)",
    )

    # 4. training_history.png (của mô hình tốt nhất)
    plot_training_history(
        best["history"],
        report_dir / "training_history.png",
        title=f"Lịch sử hàm mất mát & Độ chính xác — {best['display_name']}",
    )

    # 5. model_comparison.png (so sánh cả 3 mô hình)
    plot_model_comparison(
        models_comparison_data,
        report_dir / "model_comparison.png",
        title="So sánh hiệu năng giữa các kiến trúc học sâu (Mục 3.6 Đề cương)",
    )

    # 6. benchmark.csv (bảng kỹ thuật chuẩn)
    export_benchmark_csv(
        models_comparison_data,
        report_dir / "benchmark.csv",
    )

    # Giữ tương thích cho các script cũ
    plot_loss_curve(best["history"], report_dir, title_suffix=f"{best['display_name']} ({epochs} Epochs)")

    # Đồng bộ sang reports/ và frontend/reports/
    _sync_report_to_web(report_dir)

    logger.info(f"\n✓ Huấn luyện và tự động xuất toàn bộ 6 tệp báo cáo thành công tại: {report_dir.resolve()}")
    return {
        "best_model": best["display_name"],
        "test_f1": best["test_f1"],
        "test_acc": best["test_acc"],
        "comparison": models_comparison_data,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Huấn luyện và so sánh các kiến trúc học sâu trên GPU NVIDIA RTX 3050"
    )
    p.add_argument(
        "--data-dir", type=Path,
        default=DEFAULT_DATA_DIR,
        help="Thư mục dataset (mặc định: data/processed/dataset)",
    )
    p.add_argument(
        "--backbone", type=str,
        default="all",
        help=f"Backbone để huấn luyện: 'all' (so sánh cả 3 mô hình), hoặc: {SUPPORTED_BACKBONES} (mặc định: all)",
    )
    p.add_argument(
        "--model-dir", type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Nơi lưu mô hình (mặc định: {DEFAULT_MODEL_DIR})",
    )
    p.add_argument(
        "--report-dir", type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"Nơi lưu báo cáo (mặc định: {DEFAULT_REPORT_DIR})",
    )
    p.add_argument(
        "--epochs", type=int, default=15,
        help="Số epoch huấn luyện mỗi mô hình (mặc định: 15)",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Tốc độ học (learning rate) tùy chỉnh (mặc định: 1e-3)",
    )
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Thiết bị tính toán: 'auto' (ưu tiên CUDA RTX 3050), 'cuda', 'cpu'",
    )
    p.add_argument(
        "--no-pca", action="store_true",
        help="Tắt PCA (mặc định: bật)",
    )
    p.add_argument(
        "--force-preprocess", action="store_true",
        help="Trích xuất lại đặc trưng dù cache đã tồn tại",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Bật logging chi tiết (DEBUG)",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level  = logging.DEBUG if args.verbose else logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        results = train(
            model_dir         = args.model_dir,
            report_dir        = args.report_dir,
            data_dir          = args.data_dir,
            backbone          = args.backbone,
            use_pca           = not args.no_pca,
            force_preprocess  = args.force_preprocess,
            device            = args.device,
            epochs            = args.epochs,
            lr                = args.lr,
        )
        print(f"\n✓ Hoàn tất huấn luyện xuất sắc!")
        print(f"  Mô hình tốt nhất  : {results['best_model']}")
        print(f"  Độ chính xác Test : {results['test_acc']*100:.2f}% | Macro F1: {results['test_f1']*100:.2f}%")
        print(f"  Báo cáo lưu tại   : {args.report_dir.resolve()}")

    except RuntimeError as e:
        print(f"\n✗ Lỗi: {e}")
        raise SystemExit(1)
    except FileNotFoundError as e:
        print(f"\n✗ Lỗi: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
