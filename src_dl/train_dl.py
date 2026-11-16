"""
train_dl.py — Huấn luyện mô hình Deep Learning (CNN embedding + PyTorch MLP / sklearn).

Tính năng:
    - Huấn luyện mạng nơ-ron sâu TorchMLPClassifier trên NVIDIA CUDA GPU.
    - Ghi nhận và vẽ Sơ đồ xu hướng Loss và Accuracy theo Epoch (Train Loss vs Val Loss).
    - Hỗ trợ TIẾP TỤC HUẤN LUYỆN (Resume Training) với số Epoch tùy chỉnh (--resume --epochs N).
    - Tự động xuất biểu đồ loss_curve.png, epoch_training.png, epoch_history.csv vào reports_dl/ và reports/.
    - Lưu mô hình tốt nhất vào models_dl/classifier_dl.joblib và models_dl/config.json.

Cách dùng:
    1. Huấn luyện từ đầu (60 epochs trên CUDA):
       python -m src_dl.train_dl --models torch --epochs 60

    2. Tiếp tục huấn luyện thêm 30 epochs từ mô hình đã lưu:
       python -m src_dl.train_dl --resume --epochs 30 --lr 0.0005
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src_dl.backbone import BACKBONE_NAME, EMBEDDING_DIM
from src_dl.preprocess_dl import (
    DEFAULT_CACHE_DIR,
    extract_and_cache,
    load_cached_features,
)
from src_dl.torch_classifier import TorchMLPClassifier, plot_loss_curve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số mặc định
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR:  Path = Path("models_dl")
DEFAULT_REPORT_DIR: Path = Path("reports_dl")
FEATURE_TYPE: str = f"cnn_{BACKBONE_NAME}"

CV_FOLDS:  int = 5
N_JOBS:    int = -1          # dùng toàn bộ CPU cores cho sklearn
PCA_RATIO: float = 0.95      # giữ 95% phương sai


def _build_estimators(
    device: str = "auto",
    model_mode: str = "all",
    epochs: int = 60,
    lr: float = 1e-3,
) -> list[tuple[str, Any, dict]]:
    """Trả về danh sách (tên, estimator, param_grid)."""
    all_estimators = [
        (
            "TorchMLP_CUDA",
            TorchMLPClassifier(device=device, epochs=epochs, lr=lr, random_state=42),
            {},
        ),
        (
            "LogisticRegression",
            LogisticRegression(max_iter=2000, random_state=42, n_jobs=N_JOBS),
            {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__solver": ["lbfgs", "saga"]},
        ),
        (
            "RandomForest",
            RandomForestClassifier(random_state=42, n_jobs=N_JOBS),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10, 20]},
        ),
        (
            "SVM_RBF",
            SVC(kernel="rbf", probability=True, random_state=42),
            {"clf__C": [0.1, 1.0, 10.0, 100.0], "clf__gamma": ["scale", "auto"]},
        ),
        (
            "KNeighbors",
            KNeighborsClassifier(n_jobs=N_JOBS),
            {"clf__n_neighbors": [3, 5, 7, 11], "clf__weights": ["uniform", "distance"]},
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(random_state=42),
            {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1, 0.2]},
        ),
        (
            "DecisionTree",
            DecisionTreeClassifier(random_state=42),
            {"clf__max_depth": [None, 5, 10, 20], "clf__min_samples_split": [2, 5]},
        ),
        (
            "LDA",
            LinearDiscriminantAnalysis(),
            {"clf__solver": ["svd", "lsqr"]},
        ),
        (
            "QDA",
            QuadraticDiscriminantAnalysis(),
            {"clf__reg_param": [0.0, 0.1, 0.5]},
        ),
        (
            "GaussianNB",
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]},
        ),
    ]

    if model_mode == "torch":
        return [est for est in all_estimators if "Torch" in est[0]]
    if model_mode == "fast":
        return [est for est in all_estimators if est[0] in ("TorchMLP_CUDA", "LogisticRegression", "RandomForest")]
    if model_mode == "sklearn":
        return [est for est in all_estimators if "Torch" not in est[0]]
    return all_estimators


def _build_pipeline(estimator: Any, use_pca: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=PCA_RATIO, random_state=42)))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def _train_one(
    name: str,
    estimator: Any,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_pca: bool,
    no_gridsearch: bool,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[Any, float]:
    """Huấn luyện một bộ phân loại (TorchMLP huấn luyện trực tiếp 1 lần kèm eval_set, sklearn dùng GridSearch)."""
    pipeline = _build_pipeline(estimator, use_pca)
    is_torch_model = "Torch" in name or isinstance(estimator, TorchMLPClassifier)

    # Đối với mô hình PyTorch sâu: huấn luyện trực tiếp 1 lần duy nhất kèm theo dõi Validation Loss
    if is_torch_model:
        logger.info(f"  Huấn luyện mô hình mạng sâu {name} trực tiếp ({estimator.epochs} epochs)...")
        t0 = time.perf_counter()
        X_train_proc = X_train
        X_val_proc = X_val
        for step_name in ("scaler", "pca"):
            if step_name in pipeline.named_steps:
                transformer = pipeline.named_steps[step_name]
                X_train_proc = transformer.fit_transform(X_train_proc)
                if X_val is not None:
                    X_val_proc = transformer.transform(X_val_proc)

        eval_set = (X_val_proc, y_val) if (X_val is not None and y_val is not None) else None
        clf = pipeline.named_steps["clf"]
        clf.fit(X_train_proc, y_train, eval_set=eval_set)
        elapsed = time.perf_counter() - t0
        logger.info(f"  {name} hoàn tất trong {elapsed:.1f}s")
        return pipeline, 0.0

    if no_gridsearch or not param_grid:
        logger.info(f"  Huấn luyện {name} (không GridSearch)...")
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        logger.info(f"  {name} xong trong {elapsed:.1f}s")
        return pipeline, 0.0

    effective_n_jobs = N_JOBS
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=effective_n_jobs,
        refit=True,
        verbose=0,
    )

    logger.info(f"  GridSearchCV cho {name}...")
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        grid.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    best_score = grid.best_score_
    logger.info(
        f"  {name}: best CV F1-macro = {best_score:.4f} "
        f"| params = {grid.best_params_} | {elapsed:.1f}s"
    )
    return grid.best_estimator_, best_score


def _sync_report_to_web(report_dir: Path, web_reports_dir: Path = Path("reports")) -> None:
    """Sao chép sơ đồ loss_curve và các biểu đồ sang thư mục reports phục vụ web."""
    if not web_reports_dir.exists():
        web_reports_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("loss_curve.png", "epoch_training.png", "epoch_history.csv", "epoch_summary.json"):
        src_file = report_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, web_reports_dir / fname)
            logger.debug(f"[train_dl] Đã đồng bộ {fname} sang {web_reports_dir}")


# ---------------------------------------------------------------------------
# Chức năng: Tiếp tục huấn luyện mô hình sâu đã lưu (--resume)
# ---------------------------------------------------------------------------

def _resume_training(
    model_dir: Path,
    report_dir: Path,
    data: dict[str, Any],
    epochs: int,
    lr: float | None = None,
) -> dict[str, Any]:
    """Tiếp tục huấn luyện (resume/fine-tune) mô hình TorchMLP hiện có với số Epoch tùy chỉnh."""
    model_path = model_dir / "classifier_dl.joblib"
    config_path = model_dir / "config.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình đã lưu tại: {model_path}. "
            "Vui lòng huấn luyện từ đầu trước khi dùng --resume."
        )

    pipeline: Pipeline = joblib.load(model_path)
    old_config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            old_config = json.load(f)

    clf = pipeline.named_steps.get("clf")
    if not isinstance(clf, TorchMLPClassifier):
        raise TypeError(
            f"Mô hình đang lưu là '{type(clf).__name__}', không hỗ trợ huấn luyện nối tiếp theo epoch. "
            "Chỉ mạng nơ-ron sâu (TorchMLPClassifier) hỗ trợ tiếp tục huấn luyện."
        )

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    class_names = data["class_names"]

    # Áp dụng các bước tiền xử lý đã fit sẵn (StandardScaler, PCA)
    X_train_proc = X_train
    X_val_proc = X_val
    for step_name in ("scaler", "pca"):
        if step_name in pipeline.named_steps:
            transformer = pipeline.named_steps[step_name]
            X_train_proc = transformer.transform(X_train_proc)
            X_val_proc = transformer.transform(X_val_proc)

    # Đọc lại lịch sử cũ nếu có trong CSV
    csv_history = report_dir / "epoch_history.csv"
    if csv_history.exists() and not clf.history_:
        try:
            with open(csv_history, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                loaded_hist = []
                for row in reader:
                    loaded_hist.append({
                        "epoch": int(row["epoch"]),
                        "train_loss": float(row["train_loss"]),
                        "val_loss": float(row["val_loss"]),
                        "train_acc": float(row.get("train_acc", 0.0)),
                        "val_acc": float(row.get("val_acc", 0.0)),
                        "lr": float(row.get("lr", 1e-3)),
                    })
                clf.history_ = loaded_hist
                logger.info(f"[train_dl] Đã khôi phục lịch sử {len(loaded_hist)} epochs từ {csv_history}")
        except Exception as err:
            logger.warning(f"[train_dl] Không thể đọc lịch sử cũ: {err}")

    if not hasattr(clf, "history_") or clf.history_ is None:
        clf.history_ = []
    start_epoch = clf.history_[-1]["epoch"] if clf.history_ else 0
    logger.info(
        f"\n[train_dl] BẮT ĐẦU TIẾP TỤC HUẤN LUYỆN (Resume Training)\n"
        f"  Epoch hiện tại : {start_epoch}\n"
        f"  Số Epoch thêm  : {epochs} (Sẽ đạt tổng: {start_epoch + epochs})\n"
        f"  Learning Rate  : {lr if lr is not None else clf.lr}\n"
    )

    clf.warm_start = True
    clf.epochs = epochs
    if lr is not None:
        clf.lr = lr

    # Tiếp tục huấn luyện với tập validation để theo dõi Loss qua từng epoch
    clf.fit(X_train_proc, y_train, eval_set=(X_val_proc, y_val))

    # Vẽ và lưu sơ đồ xu hướng Loss cập nhật liên tục
    plot_loss_curve(
        clf.history_,
        report_dir,
        title_suffix=f"Tiếp tục huấn luyện ({len(clf.history_)} Epochs)",
    )
    _sync_report_to_web(report_dir)

    # Đánh giá lại trên tập test
    y_pred_test = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    logger.info(f"[train_dl] Kết quả Test sau khi tiếp tục train: Acc={test_acc:.4f} | F1={test_f1:.4f}")

    # Cập nhật pipeline đã lưu
    joblib.dump(pipeline, model_path)
    logger.info(f"[train_dl] Đã lưu mô hình cập nhật: {model_path}")

    # Cập nhật config.json
    old_config.update({
        "total_epochs": len(clf.history_),
        "test_accuracy": round(test_acc, 6),
        "test_f1": round(test_f1, 6),
        "test_metrics": {
            "accuracy": round(test_acc, 6),
            "macro_f1": round(test_f1, 6),
        },
        "last_trained_epochs": epochs,
    })
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(old_config, f, ensure_ascii=False, indent=2)

    return {
        "best_model": old_config.get("best_model", "TorchMLP_CUDA"),
        "total_epochs": len(clf.history_),
        "test_accuracy": test_acc,
        "test_f1": test_f1,
    }


# ---------------------------------------------------------------------------
# Hàm chính: huấn luyện toàn bộ pipeline
# ---------------------------------------------------------------------------

def train(
    model_dir:        Path = DEFAULT_MODEL_DIR,
    report_dir:       Path = DEFAULT_REPORT_DIR,
    cache_dir:        Path = DEFAULT_CACHE_DIR,
    data_dir:         Path = Path("data/processed/dataset"),
    use_pca:          bool = True,
    no_gridsearch:    bool = False,
    force_preprocess: bool = False,
    top_k:            int  = 3,
    device:           str  = "auto",
    model_mode:       str  = "all",
    epochs:           int  = 60,
    lr:               float | None = None,
    resume:           bool = False,
) -> dict[str, Any]:
    """Chạy toàn bộ pipeline huấn luyện DL hoặc tiếp tục huấn luyện."""
    model_dir  = Path(model_dir)
    report_dir = Path(report_dir)
    cache_dir  = Path(cache_dir)

    # Kiểm tra CUDA
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("[train_dl] Bản PyTorch hiện tại chưa hỗ trợ CUDA, tự chuyển về CPU.")
            device = "cpu"
    except ImportError:
        device = "cpu"

    # ---- Bước 1: Đảm bảo cache embedding tồn tại ----
    from src_dl.preprocess_dl import cache_is_valid

    if force_preprocess or not cache_is_valid(cache_dir, data_dir, BACKBONE_NAME):
        logger.info("[train_dl] Chạy preprocess_dl để tạo CNN embedding cache...")
        extract_and_cache(
            data_dir      = data_dir,
            cache_dir     = cache_dir,
            backbone_name = BACKBONE_NAME,
            force         = force_preprocess,
            device        = device,
        )

    # ---- Bước 2: Tải dữ liệu ----
    logger.info("[train_dl] Tải CNN embedding từ cache...")
    data        = load_cached_features(cache_dir)
    X_train     = data["X_train"]
    y_train     = data["y_train"]
    X_val       = data["X_val"]
    y_val       = data["y_val"]
    X_test      = data["X_test"]
    y_test      = data["y_test"]
    class_names = data["class_names"]

    logger.info(
        f"[train_dl] Tải xong: "
        f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}, "
        f"classes={class_names}"
    )

    # ---- Bước 3: Nếu yêu cầu tiếp tục huấn luyện (--resume) ----
    if resume:
        return _resume_training(
            model_dir  = model_dir,
            report_dir = report_dir,
            data       = data,
            epochs     = epochs,
            lr         = lr,
        )

    # ---- Bước 4: Huấn luyện từ đầu ----
    effective_lr = lr if lr is not None else 1e-3
    estimators = _build_estimators(device=device, model_mode=model_mode, epochs=epochs, lr=effective_lr)
    results: list[dict] = []

    logger.info(f"\n[train_dl] Bắt đầu huấn luyện {len(estimators)} mô hình (chế độ: {model_mode})...")
    logger.info(f"  epochs={epochs}, lr={effective_lr}, device={device}, use_pca={use_pca}, cv={CV_FOLDS}")

    for name, est, param_grid in estimators:
        try:
            best_pipeline, cv_score = _train_one(
                name, est, param_grid,
                X_train, y_train,
                use_pca, no_gridsearch,
                X_val=X_val,
                y_val=y_val,
            )

            # Đánh giá trên val
            if X_val.shape[0] > 0:
                y_pred_val = best_pipeline.predict(X_val)
                val_acc    = accuracy_score(y_val, y_pred_val)
                val_f1     = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
            else:
                val_acc = val_f1 = 0.0

            results.append({
                "name":      name,
                "pipeline":  best_pipeline,
                "cv_score":  cv_score,
                "val_acc":   val_acc,
                "val_f1":    val_f1,
            })
            logger.info(f"  {name}: val_acc={val_acc:.4f}, val_F1={val_f1:.4f}")

        except Exception as exc:
            logger.error(f"  [LỖI] {name} thất bại: {exc}", exc_info=True)

    if not results:
        raise RuntimeError("Không có mô hình nào huấn luyện thành công.")

    # Chọn top-k mô hình
    results.sort(key=lambda r: r["val_f1"], reverse=True)
    top_results = results[:top_k]

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    best = top_results[0]
    best_pipeline = best["pipeline"]

    # Nếu mô hình tốt nhất là TorchMLP, xuất sơ đồ Loss từ history_ đã ghi nhận trong quá trình huấn luyện
    clf = best_pipeline.named_steps.get("clf")
    if isinstance(clf, TorchMLPClassifier) and getattr(clf, "history_", None):
        logger.info("[train_dl] Xuất sơ đồ xu hướng Loss và Accuracy qua từng epoch...")
        plot_loss_curve(clf.history_, report_dir, title_suffix=f"{best['name']} ({epochs} Epochs)")
        _sync_report_to_web(report_dir)

    # Lưu pipeline tốt nhất
    joblib.dump(best_pipeline, model_dir / "classifier_dl.joblib")
    logger.info(f"[train_dl] Đã lưu mô hình tốt nhất: {model_dir / 'classifier_dl.joblib'}")

    # Đánh giá trên test
    y_pred_test = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

    # Lưu config.json
    config = {
        "feature_type":   FEATURE_TYPE,
        "backbone":       BACKBONE_NAME,
        "embedding_dim":  EMBEDDING_DIM,
        "crop_size":      [224, 224],
        "class_names":    class_names,
        "n_classes":      len(class_names),
        "best_model":     best["name"],
        "best_val_f1":    round(best["val_f1"], 6),
        "best_val_acc":   round(best["val_acc"], 6),
        "test_accuracy":  round(test_acc, 6),
        "test_f1":        round(test_f1, 6),
        "test_metrics": {
            "accuracy": round(test_acc, 6),
            "macro_f1": round(test_f1, 6),
        },
        "total_epochs":   epochs,
        "n_features":     EMBEDDING_DIM,
        "use_pca":        use_pca,
        "pca_ratio":      PCA_RATIO if use_pca else None,
        "cv_folds":       CV_FOLDS,
        "top_k_trained":  top_k,
        "device":         device,
    }
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Lưu báo cáo text
    _save_report(
        results       = top_results,
        X_test        = X_test,
        y_test        = y_test,
        class_names   = class_names,
        report_dir    = report_dir,
        config        = config,
    )

    logger.info(f"\n[train_dl] Hoàn tất. Mô hình: {model_dir} | Báo cáo: {report_dir}")
    return {r["name"]: {"val_f1": r["val_f1"], "val_acc": r["val_acc"]} for r in results}


def _save_report(
    results:    list[dict],
    X_test:     np.ndarray,
    y_test:     np.ndarray,
    class_names: list[str],
    report_dir: Path,
    config:     dict,
) -> None:
    """Lưu báo cáo text và JSON cho các mô hình top-k."""
    summary_lines = [
        "=" * 70,
        "BÁO CÁO HUẤN LUYỆN — src_dl (CNN Embedding + PyTorch / sklearn)",
        "=" * 70,
        f"Backbone: {config['backbone']} | Embedding dim: {config['embedding_dim']}",
        f"Feature type: {config['feature_type']}",
        f"Mô hình tốt nhất: {config['best_model']}",
        f"Tổng Epochs: {config.get('total_epochs', 'N/A')}",
        "",
        "--- Kết quả Val (sắp xếp theo F1-macro) ---",
    ]

    for i, r in enumerate(results, 1):
        summary_lines.append(
            f"  {i:2d}. {r['name']:<25s}  val_F1={r['val_f1']:.4f}  val_acc={r['val_acc']:.4f}"
        )

    if X_test.shape[0] > 0:
        summary_lines += ["", "--- Kết quả Test (mô hình tốt nhất) ---"]
        best_pipe = results[0]["pipeline"]
        y_pred_test = best_pipe.predict(X_test)
        test_acc    = accuracy_score(y_test, y_pred_test)
        test_f1     = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
        summary_lines.append(f"  Mô hình: {results[0]['name']}")
        summary_lines.append(f"  Test accuracy: {test_acc:.4f}")
        summary_lines.append(f"  Test F1-macro: {test_f1:.4f}")
        summary_lines.append("")
        summary_lines.append("  Classification Report:")
        cr = classification_report(
            y_test, y_pred_test,
            target_names=class_names,
            zero_division=0,
        )
        summary_lines.append(cr)

    report_txt = report_dir / "summary_dl.txt"
    report_txt.write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info(f"[train_dl] Đã lưu báo cáo: {report_txt}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Huấn luyện mô hình CNN + PyTorch MLP trên GPU với sơ đồ Loss theo Epoch"
    )
    p.add_argument(
        "--data-dir", type=Path,
        default=Path("data/processed/dataset"),
        help="Thư mục dataset gốc (mặc định: data/processed/dataset)",
    )
    p.add_argument(
        "--cache-dir", type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache CNN embedding (mặc định: {DEFAULT_CACHE_DIR})",
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
        "--epochs", type=int, default=60,
        help="Số epoch huấn luyện tùy chỉnh (mặc định: 60)",
    )
    p.add_argument(
        "--lr", type=float, default=None,
        help="Tốc độ học (learning rate) tùy chỉnh (mặc định: 1e-3)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Tiếp tục huấn luyện từ mô hình đã lưu với số epoch tùy chỉnh",
    )
    p.add_argument(
        "--no-pca", action="store_true",
        help="Tắt PCA (mặc định: bật)",
    )
    p.add_argument(
        "--no-gridsearch", action="store_true",
        help="Bỏ qua GridSearch (nhanh hơn, không tối ưu)",
    )
    p.add_argument(
        "--force-preprocess", action="store_true",
        help="Trích xuất lại CNN embedding dù cache đã tồn tại",
    )
    p.add_argument(
        "--top-k", type=int, default=3,
        help="Số mô hình top lưu báo cáo chi tiết (mặc định: 3)",
    )
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Thiết bị tính toán: 'auto' (ưu tiên CUDA RTX 3050), 'cuda', 'cpu'",
    )
    p.add_argument(
        "--models", type=str, default="torch", choices=["all", "torch", "fast", "sklearn"],
        help="Tập mô hình: 'torch' (PyTorch MLP trên GPU với Loss curve), 'all', 'fast', 'sklearn'",
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
            cache_dir         = args.cache_dir,
            data_dir          = args.data_dir,
            use_pca           = not args.no_pca,
            no_gridsearch     = args.no_gridsearch,
            force_preprocess  = args.force_preprocess,
            top_k             = args.top_k,
            device            = args.device,
            model_mode        = args.models,
            epochs            = args.epochs,
            lr                = args.lr,
            resume            = args.resume,
        )
        mode_str = "Tiếp tục huấn luyện" if args.resume else "Huấn luyện mới"
        print(f"\n✓ {mode_str} hoàn tất thành công.")
        print(f"  Mô hình lưu tại : {args.model_dir.resolve()}")
        print(f"  Sơ đồ Loss lưu tại: {(args.report_dir / 'loss_curve.png').resolve()}")

    except RuntimeError as e:
        print(f"\n✗ Lỗi: {e}")
        raise SystemExit(1)
    except FileNotFoundError as e:
        print(f"\n✗ Lỗi: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
