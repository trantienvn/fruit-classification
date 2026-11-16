"""
train_dl.py — Huấn luyện mô hình sklearn trên CNN embedding.

Pipeline:
    Load dl_dataset.npz (CNN embeddings đã trích xuất)
        → StandardScaler
        → PCA (tùy chọn, giữ 95% phương sai)
        → GridSearchCV trên 9 bộ phân loại sklearn
        → Lưu mô hình tốt nhất vào models_dl/

Lưu ý:
    - Augmentation ĐÃ được thực hiện trong preprocess_dl.py trên ảnh gốc.
    - File này KHÔNG tải ảnh, chỉ làm việc với vector embedding.
    - Cấu trúc 9 bộ phân loại sao chép từ src/train.py (KHÔNG import từ src/).

Chạy:
    python -m src_dl.train_dl [--model-dir PATH] [--cache-dir PATH] [--no-pca]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

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

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib  # type: ignore

from src_dl.backbone import BACKBONE_NAME, EMBEDDING_DIM
from src_dl.preprocess_dl import (
    DEFAULT_CACHE_DIR,
    extract_and_cache,
    load_cached_features,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR:  Path = Path("models_dl")
DEFAULT_REPORT_DIR: Path = Path("reports_dl")
FEATURE_TYPE: str = f"cnn_{BACKBONE_NAME}"

CV_FOLDS:  int = 5
N_JOBS:    int = -1          # dùng toàn bộ CPU cores
PCA_RATIO: float = 0.95      # giữ 95% phương sai

# ---------------------------------------------------------------------------
# Định nghĩa 9 bộ phân loại + lưới siêu tham số
# (Sao chép cấu trúc từ src/train.py — KHÔNG import từ src/)
# ---------------------------------------------------------------------------

def _build_estimators() -> list[tuple[str, Any, dict]]:
    """
    Trả về danh sách (tên, estimator, param_grid).

    9 bộ phân loại: LR, SVM, KNN, RF, GB, DT, LDA, QDA, GNB.
    """
    return [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=2000, random_state=42, n_jobs=N_JOBS),
            {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__solver": ["lbfgs", "saga"]},
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
            "RandomForest",
            RandomForestClassifier(random_state=42, n_jobs=N_JOBS),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10, 20]},
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


# ---------------------------------------------------------------------------
# Xây dựng pipeline sklearn
# ---------------------------------------------------------------------------

def _build_pipeline(estimator: Any, use_pca: bool) -> Pipeline:
    """Tạo Pipeline: Scaler → (PCA) → Classifier."""
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=PCA_RATIO, random_state=42)))
    steps.append(("clf", estimator))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Huấn luyện một bộ phân loại
# ---------------------------------------------------------------------------

def _train_one(
    name: str,
    estimator: Any,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_pca: bool,
    no_gridsearch: bool,
) -> tuple[Any, float]:
    """
    Huấn luyện một bộ phân loại với GridSearchCV (hoặc không).

    Trả về:
        (best_pipeline, best_cv_score)
    """
    pipeline = _build_pipeline(estimator, use_pca)

    if no_gridsearch or not param_grid:
        logger.info(f"  Huấn luyện {name} (không GridSearch)...")
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        logger.info(f"  {name} xong trong {elapsed:.1f}s")
        return pipeline, 0.0

    # GridSearchCV — prefix tham số với "clf__"
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=N_JOBS,
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


# ---------------------------------------------------------------------------
# Hàm chính: huấn luyện toàn bộ pipeline
# ---------------------------------------------------------------------------

def train(
    model_dir:      Path = DEFAULT_MODEL_DIR,
    report_dir:     Path = DEFAULT_REPORT_DIR,
    cache_dir:      Path = DEFAULT_CACHE_DIR,
    data_dir:       Path = Path("data/processed/dataset"),
    use_pca:        bool = True,
    no_gridsearch:  bool = False,
    force_preprocess: bool = False,
    top_k:          int  = 3,
) -> dict[str, Any]:
    """
    Chạy toàn bộ pipeline huấn luyện DL:
        1. Kiểm tra / trích xuất CNN embedding
        2. Tải dữ liệu từ cache .npz
        3. Huấn luyện 9 bộ phân loại với GridSearch
        4. Chọn top-k mô hình tốt nhất
        5. Đánh giá trên val + test
        6. Lưu artifacts vào model_dir + report_dir

    Trả về:
        dict kết quả (tên mô hình → số liệu).
    """
    model_dir  = Path(model_dir)
    report_dir = Path(report_dir)
    cache_dir  = Path(cache_dir)
    data_dir   = Path(data_dir)

    # ---- Bước 1: Đảm bảo cache tồn tại ----
    from src_dl.preprocess_dl import cache_is_valid, DEFAULT_CACHE_DIR

    if force_preprocess or not cache_is_valid(cache_dir, data_dir, BACKBONE_NAME):
        logger.info("[train_dl] Chạy preprocess_dl để tạo CNN embedding cache...")
        extract_and_cache(
            data_dir      = data_dir,
            cache_dir     = cache_dir,
            backbone_name = BACKBONE_NAME,
            force         = force_preprocess,
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

    # ---- Bước 3: Huấn luyện 9 bộ phân loại ----
    estimators = _build_estimators()
    results: list[dict] = []

    logger.info(f"\n[train_dl] Bắt đầu huấn luyện {len(estimators)} mô hình...")
    logger.info(f"  use_pca={use_pca}, no_gridsearch={no_gridsearch}, cv={CV_FOLDS}")

    for name, est, param_grid in estimators:
        try:
            best_pipeline, cv_score = _train_one(
                name, est, param_grid,
                X_train, y_train,
                use_pca, no_gridsearch,
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
            logger.error(f"  [LỖII] {name} thất bại: {exc}", exc_info=True)

    if not results:
        raise RuntimeError("Không có mô hình nào huấn luyện thành công.")

    # ---- Bước 4: Chọn top-k mô hình ----
    results.sort(key=lambda r: r["val_f1"], reverse=True)
    top_results = results[:top_k]

    logger.info(f"\n[train_dl] Top-{top_k} mô hình (theo val F1-macro):")
    for i, r in enumerate(top_results, 1):
        logger.info(f"  {i}. {r['name']}: val_F1={r['val_f1']:.4f}, val_acc={r['val_acc']:.4f}")

    # ---- Bước 5: Lưu artifacts ----
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    best = top_results[0]
    best_pipeline = best["pipeline"]

    # Lưu pipeline tốt nhất
    joblib.dump(best_pipeline, model_dir / "classifier_dl.joblib")
    logger.info(f"[train_dl] Đã lưu mô hình tốt nhất: {model_dir / 'classifier_dl.joblib'}")

    # Lưu config.json
    config = {
        "feature_type":   FEATURE_TYPE,
        "backbone":       BACKBONE_NAME,
        "embedding_dim":  EMBEDDING_DIM,
        "crop_size":      list(from_src_dl_backbone_input_size()),
        "class_names":    class_names,
        "n_classes":      len(class_names),
        "best_model":     best["name"],
        "best_val_f1":    round(best["val_f1"], 6),
        "best_val_acc":   round(best["val_acc"], 6),
        "use_pca":        use_pca,
        "pca_ratio":      PCA_RATIO if use_pca else None,
        "cv_folds":       CV_FOLDS,
        "top_k_trained":  top_k,
    }
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Lưu báo cáo text cho tất cả mô hình
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


def from_src_dl_backbone_input_size() -> tuple[int, int]:
    """Trả về INPUT_SIZE từ backbone (tránh circular import)."""
    from src_dl.backbone import INPUT_SIZE
    return INPUT_SIZE


# ---------------------------------------------------------------------------
# Lưu báo cáo
# ---------------------------------------------------------------------------

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
        "BÁO CÁO HUẤN LUYỆN — src_dl (CNN Embedding + sklearn)",
        "=" * 70,
        f"Backbone: {config['backbone']} | Embedding dim: {config['embedding_dim']}",
        f"Feature type: {config['feature_type']}",
        f"PCA: {'Có' if config['use_pca'] else 'Không'} | CV folds: {config['cv_folds']}",
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
        description="Huấn luyện mô hình sklearn trên CNN embedding (DL pipeline)"
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
        )
        print(f"\n✓ Huấn luyện hoàn tất. {len(results)} mô hình đã được đánh giá.")
        print(f"  Mô hình lưu tại: {args.model_dir.resolve()}")
        print(f"  Báo cáo lưu tại: {args.report_dir.resolve()}")

    except RuntimeError as e:
        print(f"\n✗ Lỗi: {e}")
        raise SystemExit(1)
    except FileNotFoundError as e:
        print(f"\n✗ Lỗi: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
