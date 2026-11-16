"""
train_tabular.py — Huấn luyện các mô hình Machine Learning trên bảng số liệu chất lượng trái cây.

Pipeline:
    Tabular Features (7 thuộc tính: Size, Weight, Sweetness, Crunchiness, Juiciness, Acidity, Ripeness)
        → StandardScaler
        → GridSearchCV trên các mô hình ML (RandomForest, GradientBoosting, SVM, LogisticRegression)
        → Đánh giá trên tập test
        → Lưu mô hình tốt nhất vào models_tabular/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src_tabular.dataset import (
    DEFAULT_CSV_PATH,
    FEATURE_LABELS_VI,
    FEATURE_NAMES,
    load_or_create_tabular_data,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("models_tabular")
DEFAULT_REPORT_DIR = Path("reports_tabular")
CV_FOLDS = 5


def _build_tabular_estimators() -> list[tuple[str, Any, dict]]:
    return [
        (
            "RandomForest",
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10, 20]},
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(random_state=42),
            {"clf__n_estimators": [100, 150], "clf__learning_rate": [0.05, 0.1]},
        ),
        (
            "SVM_RBF",
            SVC(kernel="rbf", probability=True, random_state=42),
            {"clf__C": [0.5, 1.0, 5.0], "clf__gamma": ["scale", "auto"]},
        ),
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1000, random_state=42),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
    ]


def train(
    csv_path: Path = DEFAULT_CSV_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    test_size: float = 0.2,
    no_gridsearch: bool = False,
) -> dict[str, Any]:
    """Huấn luyện và đánh giá các mô hình ML trên dữ liệu bảng."""
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[train_tabular] Nạp dữ liệu bảng số liệu chất lượng trái cây...")
    X, y, feature_names = load_or_create_tabular_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    logger.info(
        f"[train_tabular] Kích thước dữ liệu: Train={X_train.shape}, Test={X_test.shape}"
    )

    estimators = _build_tabular_estimators()
    results = []

    for name, est, param_grid in estimators:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])

        t0 = time.perf_counter()
        if no_gridsearch:
            pipe.fit(X_train, y_train)
            best_pipe = pipe
            cv_score = 0.0
        else:
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
            grid = GridSearchCV(
                pipe,
                param_grid,
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)
            best_pipe = grid.best_estimator_
            cv_score = grid.best_score_

        elapsed = time.perf_counter() - t0

        y_pred = best_pipe.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="macro")

        results.append(
            {
                "name": name,
                "pipeline": best_pipe,
                "cv_score": cv_score,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "train_time": elapsed,
            }
        )
        logger.info(
            f"  {name:<20s} Test Acc={test_acc:.4f} | F1={test_f1:.4f} ({elapsed:.1f}s)"
        )

    # Chọn mô hình tốt nhất
    results.sort(key=lambda r: r["test_f1"], reverse=True)
    best = results[0]
    best_pipe = best["pipeline"]

    # Lưu mô hình
    joblib.dump(best_pipe, model_dir / "classifier_tabular.joblib")
    logger.info(f"[train_tabular] Đã lưu mô hình tốt nhất: {model_dir / 'classifier_tabular.joblib'}")

    # Trích xuất độ quan trọng của đặc trưng nếu có
    clf_step = best_pipe.named_steps["clf"]
    feature_importance = {}
    if hasattr(clf_step, "feature_importances_"):
        fi = clf_step.feature_importances_
        feature_importance = {name: float(val) for name, val in zip(feature_names, fi)}
    elif hasattr(clf_step, "coef_"):
        fi = np.abs(clf_step.coef_[0])
        feature_importance = {name: float(val) for name, val in zip(feature_names, fi)}

    config = {
        "model_type": "tabular_machine_learning",
        "best_model": best["name"],
        "n_features": len(feature_names),
        "features": feature_names,
        "feature_labels_vi": FEATURE_LABELS_VI,
        "classes": ["Bad", "Good"],
        "class_labels_vi": {"Bad": "Kém (Không đạt)", "Good": "Tốt (Đạt chuẩn)"},
        "test_accuracy": round(best["test_acc"], 4),
        "test_f1": round(best["test_f1"], 4),
        "feature_importance": feature_importance,
    }

    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Lưu báo cáo chi tiết
    y_pred_best = best_pipe.predict(X_test)
    cr = classification_report(
        y_test, y_pred_best, target_names=["Bad", "Good"], zero_division=0
    )
    report_lines = [
        "=" * 70,
        "BÁO CÁO HUẤN LUYỆN HỌC MÁY THEO BẢNG SỐ LIỆU — CHẤT LƯỢNG TRÁI CÂY",
        "=" * 70,
        f"Mô hình tốt nhất : {best['name']}",
        f"Độ chính xác Test: {best['test_acc']:.4f}",
        f"F1-score (macro) : {best['test_f1']:.4f}",
        "",
        "--- So sánh các mô hình ---",
    ]
    for r in results:
        report_lines.append(
            f"  {r['name']:<20s} Acc: {r['test_acc']:.4f} | F1: {r['test_f1']:.4f} | Time: {r['train_time']:.1f}s"
        )
    report_lines += ["", "--- Báo cáo chi tiết theo lớp ---", cr]

    (report_dir / "summary_tabular.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    logger.info(f"[train_tabular] Đã lưu báo cáo: {report_dir / 'summary_tabular.txt'}")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình học máy trên bảng số liệu chất lượng trái cây"
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Đường dẫn đến file CSV dataset",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Thư mục lưu mô hình",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Thư mục lưu báo cáo",
    )
    parser.add_argument(
        "--no-gridsearch",
        action="store_true",
        help="Huấn luyện nhanh không cần GridSearch",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    train(
        csv_path=args.csv_path,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        no_gridsearch=args.no_gridsearch,
    )


if __name__ == "__main__":
    main()
