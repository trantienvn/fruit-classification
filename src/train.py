"""Pipeline huấn luyện CPU — phân loại 9 class kết hợp (loại × độ chín).

Quy trình:
  1. Nạp bộ dữ liệu dạng số (.npz) từ bước tiền xử lý ảnh -> vector pixel RGB [0, 1]
     (tự động chạy `src.preprocess` nếu chưa có cache hoặc dataset đã thay đổi)
  2. Chuẩn hóa với StandardScaler
  3. (Tùy chọn) Giảm chiều với PCA (mặc định giữ 95% phương sai)
  4. Huấn luyện 9 estimator scikit-learn trên CPU
  5. So sánh trên tập val
  6. GridSearchCV cho 3 mô hình tốt nhất
  7. Đánh giá cuối trên tập test
  8. Lưu model tốt nhất + preprocessing pipeline + metadata

Chạy:
    python -m src.preprocess
    python -m src.train
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.dataset import DATASET_CLASSIFICATION
from src.evaluate import evaluate_predictions, full_report
from src.preprocess import (
    DEFAULT_CACHE_DIR,
    REPRESENTATION_NAME,
    augment_training_images,
    cache_is_valid,
    images_to_numeric_matrix,
    load_preprocessed,
    numeric_vectors_to_images,
    pixel_feature_names,
    preprocess,
)

# ── Cấu hình mô hình ──────────────────────────────────────────────────────────

MODELS: dict[str, object] = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "NaiveBayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "SVM-RBF": CalibratedClassifierCV(SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=42), ensemble=False),
    "SVM-Linear": CalibratedClassifierCV(SVC(kernel="linear", C=1, class_weight="balanced", random_state=42), ensemble=False),
    "DecisionTree": DecisionTreeClassifier(max_depth=15, random_state=42, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# GridSearch params cho top-3 models
GRID_PARAMS: dict[str, dict] = {
    "KNN": {
        "model__n_neighbors": [3, 5, 7],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    },
    "LogisticRegression": {
        "model__C": [0.1, 1.0, 10.0],
    },
    "SVM-RBF": {
        "model__estimator__C": [1, 10, 50],
        "model__estimator__gamma": ["scale", "auto"],
        "model__estimator__kernel": ["rbf"],
    },
    "SVM-Linear": {
        "model__estimator__C": [0.1, 1, 10],
    },
    "RandomForest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 20, 30],
    },
    "GradientBoosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
    },
}

USE_PCA = True
PCA_VARIANCE = 0.95  # giữ 95% phương sai


# ── Hàm tiện ích ──────────────────────────────────────────────────────────────

def _default_data_dir() -> Path:
    """Trả về thư mục dữ liệu mặc định (dataset classification Cam/Chuối/Xoài)."""
    return DATASET_CLASSIFICATION


def _load_data(
    data_dir: Path,
    cache_dir: Path | None = None,
    force_preprocess: bool = False,
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    dict[str, int],
    tuple[str, ...],
    tuple[int, int],
    str,
]:
    """Nạp dữ liệu số đã tiền xử lý (.npz); tự tiền xử lý nếu cache chưa hợp lệ."""
    resolved_cache_dir = cache_dir or DEFAULT_CACHE_DIR
    if force_preprocess or not cache_is_valid(
        data_dir=data_dir, cache_dir=resolved_cache_dir
    ):
        print("Chưa có dữ liệu số hoặc dataset đã thay đổi — đang chạy tiền xử lý...")
        preprocess(data_dir=data_dir, cache_dir=resolved_cache_dir, force=True)
    else:
        print("Phát hiện bộ dữ liệu số đã qua xử lý — nạp trực tiếp từ .npz...")

    data = load_preprocessed(cache_dir=resolved_cache_dir)
    image_size_arr = data["image_size"]
    image_size = (int(image_size_arr[0]), int(image_size_arr[1]))
    class_names = tuple(str(name) for name in data["class_names"])
    dataset_mode = str(data["dataset_mode"][0]) if "dataset_mode" in data else "folder"
    duplicates = dict(data.get("duplicates_dropped", {"train": 0, "val": 0, "test": 0}))

    X_train = data["X_train"]
    X_test = data["X_test"]
    train_imgs = numeric_vectors_to_images(X_train, image_size=image_size)
    test_imgs = numeric_vectors_to_images(X_test, image_size=image_size)

    return (
        X_train,
        data["y_train"],
        data["X_val"],
        data["y_val"],
        X_test,
        data["y_test"],
        train_imgs,
        test_imgs,
        duplicates,
        class_names,
        image_size,
        dataset_mode,
    )


def _preprocess(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    use_pca: bool = USE_PCA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, PCA | None]:
    """Chuẩn hóa và tùy chọn PCA trên các split."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)
    X_te = scaler.transform(X_test)

    pca = None
    if use_pca:
        pca = PCA(n_components=PCA_VARIANCE, random_state=42)
        X_tr = pca.fit_transform(X_tr)
        X_va = pca.transform(X_va)
        X_te = pca.transform(X_te)
        print(f"PCA: {X_tr.shape[1]} components (giữ {PCA_VARIANCE*100:.0f}% phương sai)")

    return X_tr, X_va, X_te, scaler, pca


def _train_all(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    class_names: tuple[str, ...],
) -> tuple[dict[str, dict], str]:
    """Huấn luyện tất cả mô hình, trả về kết quả validation và tên model tốt nhất."""
    results: dict[str, dict] = {}
    print("\n── Huấn luyện các mô hình ──")
    for name, model in MODELS.items():
        estimator = clone(model)
        t0 = time.perf_counter()
        estimator.fit(X_tr, y_tr)  # type: ignore[union-attr]
        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = estimator.predict(X_va)  # type: ignore[union-attr]
        t_infer = time.perf_counter() - t0

        metrics = evaluate_predictions(y_va.tolist(), y_pred.tolist(), class_names=class_names)
        results[name] = {
            **metrics,
            "train_time_s": round(t_train, 3),
            "infer_time_ms": round(t_infer * 1000 / max(len(X_va), 1), 4),
        }
        print(
            f"  {name:22s}  acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['macro_f1']:.4f}  train={t_train:.1f}s"
        )

    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    print(f"\nMô hình tốt nhất (val F1): {best_name} = {results[best_name]['macro_f1']:.4f}")
    return results, best_name


def _evaluate_tuned_model(
    model: object,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_names: tuple[str, ...],
    fit_time_s: float,
    tune_time_s: float,
) -> dict:
    """Measure the tuned estimator on validation and separate fit/CV timings."""
    start = time.perf_counter()
    predictions = model.predict(X_val)  # type: ignore[union-attr]
    infer_time_ms = (time.perf_counter() - start) * 1000 / max(len(y_val), 1)
    metrics = evaluate_predictions(y_val.tolist(), predictions.tolist(), class_names=class_names)
    return {
        **metrics,
        "train_time_s": round(fit_time_s, 3),
        "infer_time_ms": round(infer_time_ms, 4),
        "tune_time_s": round(tune_time_s, 3),
    }


def _train_fruit_specific_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: tuple[str, ...],
    model_dir: Path,
    use_pca: bool,
) -> tuple[list[dict], dict[str, Pipeline]]:
    """Chọn riêng từng loại quả trên val, refit trên train và đánh giá test."""
    fruit_models: dict[str, Pipeline] = {}
    selected_models = []
    fruits = sorted({name.split()[0] for name in class_names})
    artifact_dir = model_dir / "per-fruit"

    for fruit in fruits:
        class_ids = [index for index, name in enumerate(class_names) if name.startswith(f"{fruit} ")]
        local_id = {class_id: index for index, class_id in enumerate(class_ids)}
        train_mask = np.isin(y_train, class_ids)
        val_mask = np.isin(y_val, class_ids)
        y_train_local = np.asarray([local_id[int(label)] for label in y_train[train_mask]])
        y_val_local = np.asarray([local_id[int(label)] for label in y_val[val_mask]])
        local_names = tuple(class_names[class_id] for class_id in class_ids)

        best_name = ""
        best_score = -1.0
        for name, estimator in MODELS.items():
            candidate = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=PCA_VARIANCE, random_state=42) if use_pca else "passthrough"),
                ("model", clone(estimator)),
            ])
            candidate.fit(X_train[train_mask], y_train_local)
            prediction = candidate.predict(X_val[val_mask])
            metrics = evaluate_predictions(
                y_val_local.tolist(), prediction.tolist(), class_names=local_names
            )
            if metrics["macro_f1"] > best_score:
                best_name, best_score = name, metrics["macro_f1"]

        fit_mask = np.isin(y_fit, class_ids)
        y_fit_local = np.asarray([local_id[int(label)] for label in y_fit[fit_mask]])
        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=PCA_VARIANCE, random_state=42) if use_pca else "passthrough"),
            ("model", clone(MODELS[best_name])),
        ])
        final_model.fit(X_fit[fit_mask], y_fit_local)

        fruit_models[fruit] = final_model
        selected_models.append((fruit, class_ids, local_id, local_names, best_name, best_score, final_model))

    results: list[dict] = []
    for fruit, class_ids, local_id, local_names, best_name, best_score, final_model in selected_models:
        test_mask = np.isin(y_test, class_ids)
        y_test_local = np.asarray([local_id[int(label)] for label in y_test[test_mask]])
        test_prediction = final_model.predict(X_test[test_mask])
        test_metrics = evaluate_predictions(
            y_test_local.tolist(), test_prediction.tolist(), class_names=local_names
        )
        fruit_dir = artifact_dir / fruit.lower()
        fruit_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, fruit_dir / "classifier.joblib")
        (fruit_dir / "config.json").write_text(
            json.dumps({"fruit": fruit, "class_names": list(local_names), "model_name": best_name}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        results.append({
            "scope": "per-fruit",
            "fruit": fruit,
            "model": best_name,
            "val_macro_f1": best_score,
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
        })

    return results, fruit_models


def _grid_search_best(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_pca: bool = False,
) -> tuple[object, dict]:
    """Tune trên train folds với preprocessing được fit riêng trong từng fold."""
    if name not in GRID_PARAMS:
        return MODELS[name], {}

    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_VARIANCE, random_state=42) if use_pca else "passthrough"),
        ("model", clone(MODELS[name])),
    ])

    gs = GridSearchCV(
        base_model,
        GRID_PARAMS[name],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_macro",
        n_jobs=-1,
    )
    print(f"\nGridSearch {name}...")
    t0 = time.perf_counter()
    gs.fit(X_train, y_train)
    print(f"  Xong — {time.perf_counter()-t0:.1f}s | best params: {gs.best_params_}")
    return gs.best_estimator_, {
        "cv_macro_f1": float(gs.best_score_),
        "best_params": dict(gs.best_params_),
    }


def _save_pipeline(
    model: object,
    scaler: StandardScaler,
    pca: PCA | None,
    model_dir: Path,
    model_name: str,
    val_metrics: dict,
    test_metrics: dict,
    training_meta: dict,
    class_names: tuple[str, ...],
    n_features: int,
    image_size: tuple[int, int],
) -> None:
    """Lưu toàn bộ pipeline cần thiết để inference."""
    model_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, Pipeline):
        scaler = model.named_steps["scaler"]
        fitted_pca = model.named_steps["pca"]
        pca = fitted_pca if isinstance(fitted_pca, PCA) else None

    joblib.dump(model, model_dir / "classifier.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    if pca is not None:
        joblib.dump(pca, model_dir / "pca.joblib")

    config = {
        "model_name": model_name,
        "n_classes": len(class_names),
        "class_names": list(class_names),
        "n_features": int(n_features),
        "feature_type": REPRESENTATION_NAME,
        "use_pca": pca is not None,
        "pipeline_includes_preprocessing": isinstance(model, Pipeline),
        "crop_size": [int(image_size[0]), int(image_size[1])],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        **training_meta,
    }
    (model_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nĐã lưu pipeline vào: {model_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────

def train(
    data_dir: Path | None = None,
    model_dir: Path | None = None,
    use_pca: bool = USE_PCA,
    run_gridsearch: bool = True,
    cache_dir: Path | None = None,
    force_preprocess: bool = False,
) -> dict:
    """Chạy pipeline các mô hình scikit-learn trên bộ dữ liệu số (.npz)."""
    if data_dir is None:
        data_dir = _default_data_dir()

    print("=" * 60)
    print("PIPELINE HUẤN LUYỆN — PHÂN LOẠI TRẠNG THÁI TRÁI CÂY")
    print(f"Dataset: {data_dir}")
    print("Biểu diễn đầu vào: Pixel RGB chuẩn hóa [0, 1] (dữ liệu số)")
    print("Thiết bị huấn luyện: CPU")
    print("=" * 60)

    # 1. Nạp bộ dữ liệu dạng số (.npz)
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        train_images, test_images,
        duplicate_counts,
        class_names,
        image_size,
        dataset_mode,
    ) = _load_data(
        data_dir,
        cache_dir=cache_dir,
        force_preprocess=force_preprocess,
    )

    if model_dir is None:
        model_dir = Path("models-whole-scene" if dataset_mode == "whole-scene" else "models")
    print(f"Model dir: {model_dir} (mode={dataset_mode}, shape={X_train.shape})")

    # 2. Chuẩn hóa & PCA
    X_tr, X_va, X_te, scaler, pca = _preprocess(X_train, X_val, X_test, use_pca)

    # 3. Huấn luyện tất cả mô hình
    val_results, best_name = _train_all(X_tr, y_train, X_va, y_val, class_names)
    available_models = dict(MODELS)
    use_gridsearch = run_gridsearch

    # 4. GridSearch cho top-3 mô hình tốt nhất
    if use_gridsearch:
        top3 = sorted(val_results, key=lambda n: val_results[n]["macro_f1"], reverse=True)[:3]
        top3_eligible = [n for n in top3 if n in GRID_PARAMS]
        best_f1 = val_results[best_name]["macro_f1"]
        best_tuned = None
        tuned_best_name = best_name

        for name in top3_eligible:
            tune_start = time.perf_counter()
            tuned, search_details = _grid_search_best(
                name, X_train, y_train, use_pca=use_pca
            )
            tune_time = time.perf_counter() - tune_start
            tuned_name = name + "_tuned"
            tuned_metrics = _evaluate_tuned_model(
                tuned,
                X_val,
                y_val,
                class_names,
                val_results[name]["train_time_s"],
                tune_time,
            )
            tuned_metrics.update(search_details)
            val_results[tuned_name] = tuned_metrics
            m = val_results[tuned_name]
            print(f"  {name} (tuned): val F1 = {m['macro_f1']:.4f}")
            if m["macro_f1"] > best_f1:
                best_f1 = m["macro_f1"]
                best_tuned = tuned
                tuned_best_name = name

        if best_tuned is not None:
            best_name = tuned_best_name
            available_models[best_name] = best_tuned
            print(f"\nMô hình tốt nhất sau GridSearch: {best_name} — val F1 = {best_f1:.4f}")
        else:
            print(f"\nGiữ nguyên mô hình ban đầu: {best_name} — val F1 = {best_f1:.4f}")

    # Refit pipeline trên train + augmentation (chuyển ảnh augmented thành vector số cùng chuẩn)
    selected_estimator = available_models[best_name]
    best_model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_VARIANCE, random_state=42) if use_pca else "passthrough"),
        ("model", clone(selected_estimator.named_steps["model"]) if isinstance(selected_estimator, Pipeline) else clone(selected_estimator)),
    ])
    augmented_images, augmented_labels = augment_training_images(train_images, y_train)
    augmented_vectors = images_to_numeric_matrix(
        augmented_images[len(train_images):], image_size=image_size
    )
    augmentation_samples = []
    seen_classes = set()
    for index, label in enumerate(y_train):
        class_id = int(label)
        if class_id in seen_classes:
            continue
        augmentation_samples.append((
            train_images[index],
            augmented_images[len(train_images) + index],
            class_names[class_id],
        ))
        seen_classes.add(class_id)
        if len(augmentation_samples) == min(6, len(class_names)):
            break
    X_fit = np.vstack([X_train, augmented_vectors])
    y_fit = np.concatenate([y_train, np.asarray(augmented_labels[len(y_train):])])
    best_model.fit(X_fit, y_fit)

    selected_val = evaluate_predictions(
        y_val.tolist(), best_model.predict(X_val).tolist(), class_names=class_names
    )
    per_fruit_results, per_fruit_models = _train_fruit_specific_models(
        X_train,
        y_train,
        X_val,
        y_val,
        X_fit,
        y_fit,
        X_test,
        y_test,
        class_names,
        model_dir,
        use_pca,
    )
    per_fruit_results.insert(0, {
        "scope": "global",
        "fruit": "all",
        "model": best_name,
        "val_macro_f1": selected_val["macro_f1"],
        "test_accuracy": "",
        "test_macro_f1": "",
    })

    # 5. Đánh giá tập test độc lập một lần duy nhất
    y_pred_test = best_model.predict(X_test)  # type: ignore[union-attr]
    test_metrics = evaluate_predictions(
        y_test.tolist(), y_pred_test.tolist(), class_names=class_names
    )
    per_fruit_results[0]["test_accuracy"] = test_metrics["accuracy"]
    per_fruit_results[0]["test_macro_f1"] = test_metrics["macro_f1"]
    print("\n── Kết quả test ──")
    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1 : {test_metrics['macro_f1']:.4f}")

    # 6. Xuất biểu đồ và báo cáo
    reports_dir = model_dir.parent / (
        "reports-whole-scene" if dataset_mode == "whole-scene" else "reports"
    )
    full_report(
        y_test.tolist(), y_pred_test.tolist(),
        class_names=class_names,
        model_name=best_name,
        all_val_results=val_results,
        save_dir=reports_dir,
        model=best_model,
        X_test=X_test,
        X_val=X_val,
        y_val=y_val.tolist(),
        X_train=X_train,
        y_train=y_train.tolist(),
        test_images=test_images,
        ablation_results=None,
        per_fruit_results=per_fruit_results,
        selected_val_metrics=selected_val,
        feature_names=pixel_feature_names(image_size),
        augmentation_samples=augmentation_samples,
    )

    # 7. Lưu pipeline
    val_summary = {k: {m: v for m, v in val_results[k].items() if m in ("accuracy", "macro_f1")}
                   for k in val_results}
    _save_pipeline(
        best_model, scaler, pca, model_dir, best_name,
        val_metrics=val_summary,
        test_metrics={k: test_metrics[k] for k in ("accuracy", "macro_f1")},
        class_names=class_names,
        n_features=int(X_train.shape[1]),
        image_size=image_size,
        training_meta={
            "n_train": int(len(y_train)),
            "n_train_after_augmentation": int(len(y_fit)),
            "augmentation": "1 rotation/flip/brightness variant per train image; seed=42",
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "use_pca": pca is not None,
            "gridsearch": use_gridsearch,
            "backend": "scikit-learn",
            "device": "cpu",
            "dataset_mode": dataset_mode,
            "duplicate_images_dropped": duplicate_counts,
            "per_fruit_models": sorted(per_fruit_models),
            "selected_validation_metrics": {
                "accuracy": selected_val["accuracy"],
                "macro_f1": selected_val["macro_f1"],
            },
        },
    )

    return test_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân loại trái cây")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    pca_group = parser.add_mutually_exclusive_group()
    pca_group.add_argument("--pca", dest="use_pca", action="store_true", help="Bật PCA 95%% phương sai (mặc định)")
    pca_group.add_argument("--no-pca", dest="use_pca", action="store_false", help="Tắt PCA")
    parser.set_defaults(use_pca=USE_PCA)
    parser.add_argument("--no-gridsearch", action="store_true", help="Tắt GridSearch")
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Ép chuyển đổi lại bộ dữ liệu số .npz trước khi huấn luyện",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir or _default_data_dir()
    missing_splits = [
        split for split in ("train", "val", "test")
        if not (data_dir / split).is_dir()
    ]
    if missing_splits:
        print(
            f"LỖI: Thiếu split {', '.join(missing_splits)} trong {data_dir}. "
            "Hãy kiểm tra --data-dir hoặc chạy bước chuẩn bị dataset."
        )
        return 2

    try:
        test_metrics = train(
            data_dir=data_dir,
            model_dir=args.model_dir,
            use_pca=args.use_pca,
            run_gridsearch=not args.no_gridsearch,
            cache_dir=args.cache_dir,
            force_preprocess=args.force_preprocess,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"LỖI: {exc}")
        return 2
    print(f"\nTest accuracy: {test_metrics['accuracy']:.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())