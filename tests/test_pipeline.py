"""Kiểm thử các module ML pipeline — không cần dataset thật."""

from __future__ import annotations

import io

import numpy as np
import pytest
from matplotlib.axes import Axes
from PIL import Image, ImageDraw
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import (
    CLASS_FOLDER_NAMES,
    CLASS_NAMES,
    DATA_RAW,
    DATASET_WHOLE_SCENE,
    _parse_yolo_box,
    class_distribution,
    deduplicate_splits,
    load_classification_split,
    load_scene_split,
    prepare_whole_scene_dataset,
)
from src.audit import audit_folder_dataset
from src.evaluate import (
    full_report,
    _plot_feature_importance,
    _plot_learning_curve,
    _plot_model_comparison,
    _save_misclassified_images,
    _save_model_comparison_csv,
    evaluate_predictions,
)
from src.features import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    N_FEATURES,
    augment_training_images,
    extract_features,
    save_mask_samples,
    segment_fruit,
)
from src.predict import FruitClassifier, checkpoint_available
from src.train import (
    GRID_PARAMS,
    _default_data_dir,
    _evaluate_tuned_model,
    _grid_search_best,
    _train_all,
    main,
)


# ── Dataset ──────────────────────────────────────────────────────────────────

def test_parse_yolo_box_valid():
    """Chuyển đổi dòng YOLO hợp lệ thành tọa độ pixel."""
    result = _parse_yolo_box("1 0.5 0.5 0.8 0.6", img_w=100, img_h=100)
    assert result is not None
    class_id, x1, y1, x2, y2 = result
    assert class_id == 1
    assert x1 == 10 and y1 == 20 and x2 == 90 and y2 == 80


def test_parse_yolo_box_invalid_returns_none():
    assert _parse_yolo_box("not valid", 100, 100) is None
    assert _parse_yolo_box("99 0.5 0.5 0.1 0.1", 100, 100) is None  # class out of range
    assert _parse_yolo_box("0 0.5 0.5 0.1", 100, 100) is None  # wrong fields


def test_class_distribution():
    labels = [0, 1, 1, 2, 0, 8]
    dist = class_distribution(labels)
    assert dist["Orange Unripe"] == 2
    assert dist["Orange Ripe"] == 2
    assert dist["Orange Spoiled"] == 1
    assert dist["Mango Spoiled"] == 1
    assert len(dist) == len(CLASS_NAMES)


def test_deduplicate_splits_prefers_test_then_validation():
    image_a = np.zeros((8, 8, 3), dtype=np.uint8)
    image_b = np.ones((8, 8, 3), dtype=np.uint8)

    train, val, test, dropped = deduplicate_splits(
        [image_a, image_b], [0, 1], [image_a], [0], [image_b], [1]
    )

    assert len(train[0]) == 0
    assert len(val[0]) == 1
    assert len(test[0]) == 1
    assert dropped == {"test": 0, "val": 0, "train": 2}


def test_audit_folder_dataset_reports_corrupt_and_cross_split_duplicates(tmp_path):
    for split in ("train", "val", "test"):
        for folder in CLASS_FOLDER_NAMES.values():
            (tmp_path / split / folder).mkdir(parents=True)
    shared = Image.new("RGB", (12, 8), "orange")
    shared.save(tmp_path / "train" / "cam-chin" / "train.png")
    shared.save(tmp_path / "val" / "cam-chin" / "val.png")
    (tmp_path / "test" / "cam-chin" / "broken.png").write_bytes(b"not an image")

    report = audit_folder_dataset(tmp_path)

    assert report["total_images"] == 3
    assert len(report["corrupt_images"]) == 1
    assert len(report["duplicate_groups"]) == 1
    assert len(report["cross_split_duplicate_groups"]) == 1
    assert report["image_dimensions"] == {"12x8": 2}


def test_load_classification_split_maps_folder_labels_and_resizes(tmp_path):
    split_dir = tmp_path / "train"
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / CLASS_FOLDER_NAMES[class_name]
        class_dir.mkdir(parents=True)
        Image.new("RGB", (40, 20), (class_id * 20, 50, 90)).save(class_dir / "sample.png")

    images, labels = load_classification_split(tmp_path, "train")

    assert labels == list(range(len(CLASS_NAMES)))
    assert len(images) == len(CLASS_NAMES)
    assert all(image.shape == (128, 128, 3) for image in images)


def test_prepare_whole_scene_dataset_preserves_splits_and_filters_ambiguous_images(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "whole-scene"

    for split in ("train", "val", "test"):
        image_dir = source_dir / "images" / split
        label_dir = source_dir / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for class_id, class_name in enumerate(CLASS_NAMES):
            image_name = f"sample-{class_id}.png"
            Image.new("RGB", (160, 90), (class_id * 20, 50, 90)).save(image_dir / image_name)
            (label_dir / f"sample-{class_id}.txt").write_text(
                f"{class_id} 0.5 0.5 0.5 0.5\n",
                encoding="utf-8",
            )

        if split == "train":
            Image.new("RGB", (160, 90), "white").save(image_dir / "mixed.png")
            (label_dir / "mixed.txt").write_text(
                "0 0.25 0.5 0.2 0.3\n1 0.75 0.5 0.2 0.3\n",
                encoding="utf-8",
            )
            Image.new("RGB", (160, 90), "black").save(image_dir / "unlabeled.png")
            (label_dir / "unlabeled.txt").write_text("", encoding="utf-8")
            (label_dir / "invalid.txt").write_text("invalid label\n", encoding="utf-8")
            Image.new("RGB", (160, 90), "red").save(image_dir / "invalid.png")
            Image.new("RGB", (160, 90), "green").save(image_dir / "missing-label.png")

    summary = prepare_whole_scene_dataset(source_dir, output_dir)
    train_summary = summary["splits"]["train"]
    assert train_summary["included"] == len(CLASS_NAMES)
    assert train_summary["excluded"] == {
        "missing_label": 1,
        "empty_label": 1,
        "invalid_label": 1,
        "multiple_classes": 1,
    }

    exported_image = output_dir / "train" / "Apple_Unripe" / "sample-0.png"
    assert Image.open(exported_image).size == (160, 90)
    images, labels = load_scene_split(output_dir, "train")
    assert len(images) == len(CLASS_NAMES)
    assert set(labels) == set(range(len(CLASS_NAMES)))
    assert all(image.shape == (128, 128, 3) for image in images)

    assert prepare_whole_scene_dataset(source_dir, output_dir) == summary


def test_training_default_dataset_dir():
    assert _default_data_dir() == DATASET_WHOLE_SCENE


def test_training_cli_uses_whole_scene_and_has_no_dataset_mode(monkeypatch, tmp_path, capsys):
    data_dir = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (data_dir / split).mkdir(parents=True)
    monkeypatch.setattr("src.train.DATASET_WHOLE_SCENE", data_dir)
    captured = {}

    def fake_train(**kwargs):
        captured.update(kwargs)
        return {"accuracy": 1.0}

    monkeypatch.setattr("src.train.train", fake_train)

    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    assert "--dataset-mode" not in capsys.readouterr().out

    assert main([]) == 0
    assert captured["data_dir"] == data_dir
    assert captured["use_pca"] is True
    assert "device" not in captured


def test_grid_search_tunes_calibrated_svm_estimator():
    rng = np.random.default_rng(42)
    X_train = np.vstack([
        rng.normal(loc=-1, size=(10, 3)),
        rng.normal(loc=1, size=(10, 3)),
    ])
    y_train = np.array([0] * 10 + [1] * 10)
    X_val = np.vstack([
        rng.normal(loc=-1, size=(4, 3)),
        rng.normal(loc=1, size=(4, 3)),
    ])
    y_val = np.array([0] * 4 + [1] * 4)
    model = _grid_search_best("SVM-RBF", X_train, y_train)

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(model.named_steps["model"], CalibratedClassifierCV)
    assert set(GRID_PARAMS["SVM-RBF"]) == {
        "model__estimator__C",
        "model__estimator__gamma",
        "model__estimator__kernel",
    }
    assert model.get_params()["model__estimator__kernel"] == "rbf"
    assert model.predict(X_val).shape == y_val.shape


# ── Features ─────────────────────────────────────────────────────────────────

def _make_test_image(h: int = 128, w: int = 128, color: tuple = (180, 120, 60)) -> np.ndarray:
    """Tạo ảnh RGB ngẫu nhiên có màu chủ đạo."""
    rng = np.random.default_rng(42)
    base = np.full((h, w, 3), color, dtype=np.uint8)
    noise = rng.integers(-20, 20, (h, w, 3), dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_extract_features_shape():
    img = _make_test_image()
    feat = extract_features(img)
    assert feat.shape == (N_FEATURES,), f"Expected ({N_FEATURES},), got {feat.shape}"
    assert feat.dtype == np.float64
    assert len(set(FEATURE_NAMES)) == N_FEATURES


def test_extract_features_finite():
    """Tất cả giá trị đặc trưng phải hữu hạn."""
    img = _make_test_image()
    feat = extract_features(img)
    assert np.all(np.isfinite(feat)), "Phát hiện NaN hoặc Inf trong feature vector"


def test_extract_features_different_images():
    """Hai ảnh khác nhau phải cho features khác nhau."""
    img1 = _make_test_image(color=(200, 50, 50))    # đỏ
    img2 = _make_test_image(color=(50, 200, 50))    # xanh lá
    feat1 = extract_features(img1)
    feat2 = extract_features(img2)
    assert not np.allclose(feat1, feat2)


def test_extract_features_consistent():
    """Cùng một ảnh phải cho cùng features."""
    img = _make_test_image()
    feat1 = extract_features(img)
    feat2 = extract_features(img)
    np.testing.assert_array_equal(feat1, feat2)


def test_mask_feature_groups_and_train_augmentation(tmp_path):
    canvas = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(canvas).ellipse((24, 24, 104, 104), fill=(20, 180, 30))
    image = np.asarray(canvas, dtype=np.uint8)
    mask = segment_fruit(image)
    features = extract_features(image)
    augmented, labels = augment_training_images([image], [2])

    assert 0.05 < np.count_nonzero(mask) / mask.size < 0.8
    assert features.shape == (90,)
    assert [FEATURE_GROUPS[name].stop - FEATURE_GROUPS[name].start for name in ("color", "texture", "shape")] == [70, 14, 6]
    assert len(augmented) == 2 and labels == [2, 2]
    assert save_mask_samples([image], str(tmp_path)) == 1
    assert (tmp_path / "mask_00.png").is_file()


# ── Evaluate ─────────────────────────────────────────────────────────────────

def test_evaluate_predictions_perfect():
    metrics = evaluate_predictions([0, 1, 2], [0, 1, 2])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_evaluate_predictions_partial():
    metrics = evaluate_predictions([0, 1, 2, 0], [0, 1, 2, 1])
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert "report" in metrics


def test_tuned_model_result_keeps_fit_and_tuning_times_separate():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = DummyClassifier(strategy="most_frequent").fit(X, y)

    result = _evaluate_tuned_model(model, X, y, ("Unripe", "Ripe"), 0.25, 1.5)

    assert result["train_time_s"] == 0.25
    assert result["tune_time_s"] == 1.5
    assert result["infer_time_ms"] >= 0


def test_evaluate_predictions_with_class_names():
    metrics = evaluate_predictions(
        [0, 1], [0, 0],
        class_names=["Apple Unripe", "Apple Ripe"]
    )
    assert "Apple Unripe" in metrics["report"]


def test_model_comparison_report_includes_logistic_regression(tmp_path, monkeypatch):
    plotted_names = []
    set_xticklabels = Axes.set_xticklabels

    def capture_xticklabels(axis, labels, *args, **kwargs):
        plotted_names.extend(str(label) for label in labels)
        return set_xticklabels(axis, labels, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_xticklabels", capture_xticklabels)
    monkeypatch.setattr(
        "src.train.MODELS",
        {"LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)},
    )
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    results, best_name = _train_all(X, y, X, y, ("Unripe", "Ripe"))
    output_path = tmp_path / "model_comparison.png"

    _plot_model_comparison(results, output_path)

    assert best_name == "LogisticRegression"
    assert results["LogisticRegression"]["accuracy"] == 1
    assert "LogisticRegression" in plotted_names
    assert output_path.is_file()
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_model_comparison_csv_records_validation_metrics(tmp_path):
    output_path = tmp_path / "model_comparison.csv"
    _save_model_comparison_csv(
        {"Dummy": {"accuracy": 0.5, "macro_f1": 0.3, "train_time_s": 0.01}},
        output_path,
    )

    text = output_path.read_text(encoding="utf-8-sig")
    assert "Dummy" in text
    assert "0.5" in text
    assert "0.3" in text


def test_learning_curve_and_misclassified_images_are_saved(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 2))
    y = np.array([0] * 15 + [1] * 15)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]).fit(X, y)
    learning_path = tmp_path / "learning_curve.png"

    _plot_learning_curve(model, X, y, learning_path)
    _save_misclassified_images(
        [np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)],
        np.array([0, 1]),
        np.array([1, 1]),
        ["Orange Unripe", "Orange Ripe"],
        tmp_path,
    )

    assert learning_path.is_file()
    assert (tmp_path / "misclassified.csv").is_file()
    assert len(list((tmp_path / "misclassified").glob("*.png"))) == 1


def test_full_report_writes_required_metrics_and_plots(tmp_path):
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(45, N_FEATURES))
    y_train = np.repeat(np.arange(3), 15)
    X_val = rng.normal(size=(9, N_FEATURES))
    y_val = np.repeat(np.arange(3), 3)
    X_test = rng.normal(size=(9, N_FEATURES))
    y_test = np.repeat(np.arange(3), 3)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    class_names = ["Orange Unripe", "Orange Ripe", "Orange Spoiled"]

    full_report(
        y_test.tolist(), y_pred.tolist(), class_names, "LogisticRegression",
        {"LogisticRegression": {"accuracy": 0.5, "macro_f1": 0.4}},
        tmp_path, model=model, X_test=X_test, X_val=X_val, y_val=y_val.tolist(),
        X_train=X_train, y_train=y_train.tolist(),
        test_images=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in y_test],
        ablation_results={"color": {"best_model": "LR", "accuracy": 0.5, "macro_f1": 0.4}},
        per_fruit_results=[{"scope": "global", "fruit": "all", "model": "LR", "val_macro_f1": 0.4, "test_accuracy": 0.5, "test_macro_f1": 0.4}],
    )

    for filename in (
        "confusion_matrix.png", "roc_curves.png", "roc_auc.csv",
        "model_comparison.csv", "learning_curve.png", "learning_curve.csv",
        "classification_metrics.csv", "summary.md", "ablation_comparison.csv",
        "per_fruit_comparison.csv", "misclassified.csv",
    ):
        assert (tmp_path / filename).is_file(), filename


def test_feature_importance_uses_permutation_for_models_without_builtin_importance(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, N_FEATURES))
    y = np.array([0, 1] * 20)
    model = DummyClassifier(strategy="most_frequent").fit(X, y)
    output_path = tmp_path / "feature_importance.png"

    generated = _plot_feature_importance(
        model,
        FEATURE_NAMES,
        output_path,
        X_eval=X,
        y_eval=y,
    )

    assert generated
    assert output_path.is_file()
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


# ── Predict ──────────────────────────────────────────────────────────────────

def test_checkpoint_available_false(tmp_path):
    """Trả về False khi không có model."""
    assert not checkpoint_available(tmp_path / "nonexistent")


def test_model_exists_false(tmp_path):
    assert not FruitClassifier.model_exists(tmp_path)


def test_classifier_load_raises_when_missing(tmp_path):
    """Raise ModelNotReadyError khi thiếu file model."""
    from src.predict import ModelNotReadyError
    clf = FruitClassifier(tmp_path)
    with pytest.raises(ModelNotReadyError):
        clf.load()


def test_classifier_predict_raises_when_not_loaded(tmp_path):
    """Raise ModelNotReadyError khi chưa load."""
    from src.predict import ModelNotReadyError
    clf = FruitClassifier(tmp_path)
    img = Image.new("RGB", (64, 64), "green")
    with pytest.raises(ModelNotReadyError):
        clf.predict(img)


# ── Require dataset ───────────────────────────────────────────────────────────

def test_require_dataset_reports_missing_data(tmp_path):
    """Báo lỗi khi thư mục data không có ảnh."""
    from src.data import find_images
    assert find_images(tmp_path / "missing") == []
