"""Pipeline huấn luyện CPU — phân loại 9 class kết hợp (loại × độ chín).

Quy trình:
  1. Nạp ảnh nguyên cảnh từ data/processed/whole-scene/ theo các split có sẵn
  2. Trích xuất đặc trưng (51 features)
  3. Chuẩn hóa với StandardScaler
  4. (Tùy chọn) Giảm chiều với PCA
  5. Huấn luyện 7 mô hình: KNN, NaiveBayes, LR, SVM, DT, RF, GBM
  6. So sánh trên tập val
  7. GridSearchCV cho 3 mô hình tốt nhất
  8. Đánh giá cuối trên tập test
  9. Lưu model tốt nhất + preprocessing pipeline + metadata

Chạy:
  python -m src.train --data-dir data/processed/whole-scene
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.dataset import (
    CLASS_NAMES,
    DATA_RAW,
    DATASET_CLASSIFICATION,
    DATASET_WHOLE_SCENE,
    LEGACY_CLASS_NAMES,
    deduplicate_splits,
    load_classification_split,
    load_scene_split,
    load_split,
)
from src.evaluate import evaluate_predictions, full_report
from src.features import (
    FEATURE_GROUPS,
    N_FEATURES,
    augment_training_images,
    build_feature_matrix,
    save_mask_samples,
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

# Dùng PCA nếu True (tắt mặc định — thực nghiệm xác nhận)
USE_PCA = True
PCA_VARIANCE = 0.95  # giữ 95% phương sai


# ── Hàm tiện ích ──────────────────────────────────────────────────────────────

def _default_data_dir(dataset_mode: str) -> Path:
    """Trả về thư mục dữ liệu mặc định tương ứng với chế độ."""
    return {
        "folder": DATASET_CLASSIFICATION,
        "yolo": DATA_RAW,
        "whole-scene": DATASET_WHOLE_SCENE,
    }[dataset_mode]


def _load_data(data_dir: Path, dataset_mode: str) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    dict[str, int],
]:
    """Nạp và trích xuất features từ ba split."""
    split_loader = {
        "folder": load_classification_split,
        "whole-scene": load_scene_split,
        "yolo": load_split,
    }[dataset_mode]
    print("Đang nạp tập train...")
    t0 = time.perf_counter()
    train_imgs, y_train = split_loader(data_dir, "train")
    print(f"  {len(y_train)} mẫu — {time.perf_counter()-t0:.1f}s")

    print("Đang nạp tập val...")
    t0 = time.perf_counter()
    val_imgs, y_val = split_loader(data_dir, "val")
    print(f"  {len(y_val)} mẫu — {time.perf_counter()-t0:.1f}s")

    print("Đang nạp tập test...")
    t0 = time.perf_counter()
    test_imgs, y_test = split_loader(data_dir, "test")
    print(f"  {len(y_test)} mẫu — {time.perf_counter()-t0:.1f}s")

    (train_imgs, y_train), (val_imgs, y_val), (test_imgs, y_test), duplicates = deduplicate_splits(
        train_imgs, y_train, val_imgs, y_val, test_imgs, y_test
    )
    dropped = sum(duplicates.values())
    if dropped:
        print(f"  Loại {dropped} ảnh trùng chính xác (test > val > train): {duplicates}")

    print("\nTrích xuất đặc trưng...")
    t0 = time.perf_counter()
    X_train = build_feature_matrix(train_imgs)
    X_val = build_feature_matrix(val_imgs)
    X_test = build_feature_matrix(test_imgs)
    print(f"  Xong — {time.perf_counter()-t0:.1f}s | shape train: {X_train.shape}")

    return (
        X_train,
        np.array(y_train),
        X_val,
        np.array(y_val),
        X_test,
        np.array(y_test),
        train_imgs,
        test_imgs,
        duplicates,
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
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)  # type: ignore[union-attr]
        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(X_va)  # type: ignore[union-attr]
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


def _run_feature_ablations(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_names: tuple[str, ...],
    use_pca: bool,
) -> dict[str, dict]:
    """Chọn mô hình theo validation cho màu, màu+kết cấu và toàn bộ feature."""
    variants = {
        "color": ("color",),
        "color+texture": ("color", "texture"),
        "color+texture+shape": ("color", "texture", "shape"),
    }
    results = {}
    for variant, groups in variants.items():
        columns = np.concatenate([
            np.arange(FEATURE_GROUPS[group].start, FEATURE_GROUPS[group].stop)
            for group in groups
        ])
        X_train_variant = X_train[:, columns]
        X_val_variant = X_val[:, columns]
        X_tr, X_va, _, _, _ = _preprocess(
            X_train_variant, X_val_variant, X_val_variant, use_pca
        )
        model_results, best_name = _train_all(
            X_tr, y_train, X_va, y_val, class_names
        )
        best = model_results[best_name]
        results[variant] = {
            "best_model": best_name,
            "accuracy": best["accuracy"],
            "macro_f1": best["macro_f1"],
        }
    return results


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
) -> object:
    """Tune trên train folds với preprocessing được fit riêng trong từng fold."""
    if name not in GRID_PARAMS:
        return MODELS[name]

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
    return gs.best_estimator_


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
        "n_features": N_FEATURES,
        "use_pca": pca is not None,
        "pipeline_includes_preprocessing": isinstance(model, Pipeline),
        "crop_size": [128, 128],
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
    dataset_mode: str = "folder",
) -> dict:
    """Chạy pipeline các mô hình scikit-learn trên CPU."""
    if dataset_mode not in {"folder", "yolo", "whole-scene"}:
        raise ValueError("dataset_mode phải là 'folder', 'yolo' hoặc 'whole-scene'.")
    if data_dir is None:
        data_dir = _default_data_dir(dataset_mode)
    if model_dir is None:
        model_dir = Path(
            "models-whole-scene" if dataset_mode == "whole-scene" else "models"
        )
    class_names = CLASS_NAMES if dataset_mode == "folder" else LEGACY_CLASS_NAMES
    print("=" * 60)
    print("PIPELINE HUẤN LUYỆN — PHÂN LOẠI TRẠNG THÁI TRÁI CÂY")
    print(f"Dataset: {data_dir}")
    print(f"Chế độ dữ liệu: {dataset_mode}")
    print(f"Model dir: {model_dir}")
    print("Thiết bị huấn luyện: CPU")
    print("=" * 60)

    # 1. Nạp và trích xuất features
    X_train, y_train, X_val, y_val, X_test, y_test, train_images, test_images, duplicate_counts = _load_data(data_dir, dataset_mode)

    # 2. Chuẩn hóa
    X_tr, X_va, X_te, scaler, pca = _preprocess(X_train, X_val, X_test, use_pca)

    # 3. So sánh nhóm đặc trưng chỉ trên train/validation.
    ablation_results = _run_feature_ablations(
        X_train, y_train, X_val, y_val, class_names, use_pca
    )

    # 4. Huấn luyện tất cả mô hình
    val_results, best_name = _train_all(X_tr, y_train, X_va, y_val, class_names)
    available_models = MODELS
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
            tuned = _grid_search_best(name, X_train, y_train, use_pca=use_pca)
            tune_time = time.perf_counter() - tune_start
            tuned_name = name + "_tuned"
            val_results[tuned_name] = _evaluate_tuned_model(
                tuned,
                X_val,
                y_val,
                class_names,
                val_results[name]["train_time_s"],
                tune_time,
            )
            m = val_results[tuned_name]
            print(f"  {name} (tuned): val F1 = {m['macro_f1']:.4f}")
            if m["macro_f1"] > best_f1:
                best_f1 = m["macro_f1"]
                best_tuned = tuned
                tuned_best_name = name

        if best_tuned is not None:
            best_name = tuned_best_name
            available_models[best_name] = best_tuned  # type: ignore[assignment]
            print(f"\nMô hình tốt nhất sau GridSearch: {best_name} — val F1 = {best_f1:.4f}")
        else:
            print(f"\nGiữ nguyên mô hình ban đầu: {best_name} — val F1 = {best_f1:.4f}")

    # Refit a single inference pipeline on train only, regardless of model family.
    selected_estimator = available_models[best_name]
    best_model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_VARIANCE, random_state=42) if use_pca else "passthrough"),
        ("model", clone(selected_estimator.named_steps["model"]) if isinstance(selected_estimator, Pipeline) else clone(selected_estimator)),
    ])
    augmented_images, augmented_labels = augment_training_images(train_images, y_train)
    augmented_features = build_feature_matrix(augmented_images[len(train_images):])
    X_fit = np.vstack([X_train, augmented_features])
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

    # 5. Evaluate the held-out test split once, after model selection is complete.
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
    if dataset_mode == "whole-scene":
        reports_dir = model_dir.parent / "reports-whole-scene"
    else:
        reports_dir = model_dir.parent / "reports"
    save_mask_samples(train_images, str(reports_dir / "mask_samples"), y_train)
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
        ablation_results=ablation_results,
        per_fruit_results=per_fruit_results,
        selected_val_metrics=selected_val,
    )

    # 7. Lưu pipeline
    val_summary = {k: {m: v for m, v in val_results[k].items() if m in ("accuracy", "macro_f1")}
                   for k in val_results}
    _save_pipeline(
        best_model, scaler, pca, model_dir, best_name,
        val_metrics=val_summary,
        test_metrics={k: test_metrics[k] for k in ("accuracy", "macro_f1")},
        class_names=class_names,
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
    parser.add_argument(
        "--dataset-mode",
        choices=("folder", "yolo", "whole-scene"),
        default="folder",
        help=(
            "whole-scene: ảnh nguyên cảnh ; "
            "folder: dataset classification theo thư mục (mặc định); yolo: crop bbox YOLO"
        ),
    )
    pca_group = parser.add_mutually_exclusive_group()
    pca_group.add_argument("--pca", dest="use_pca", action="store_true", help="Bật PCA 95%% phương sai (mặc định)")
    pca_group.add_argument("--no-pca", dest="use_pca", action="store_false", help="Tắt PCA")
    parser.set_defaults(use_pca=USE_PCA)
    parser.add_argument("--no-gridsearch", action="store_true", help="Tắt GridSearch")
    args = parser.parse_args(argv)

    data_dir = args.data_dir or _default_data_dir(args.dataset_mode)
    image_root = data_dir / "images" if args.dataset_mode == "yolo" else data_dir
    missing_splits = [
        split for split in ("train", "val", "test")
        if not (image_root / split).is_dir()
    ]
    if missing_splits:
        print(
            f"LỖI: Thiếu split {', '.join(missing_splits)} trong {image_root}. "
            "Hãy kiểm tra --data-dir hoặc chạy bước chuẩn bị dataset."
        )
        return 2

    try:
        test_metrics = train(
            data_dir=data_dir,
            model_dir=args.model_dir,
            use_pca=args.use_pca,
            run_gridsearch=not args.no_gridsearch,
            dataset_mode=args.dataset_mode,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"LỖI: {exc}")
        return 2
    print(f"\nTest accuracy: {test_metrics['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
