"""Kiểm thử các module ML pipeline — không cần dataset thật."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sklearn.dummy import DummyClassifier

from src.dataset import (
    CLASS_NAMES,
    _parse_yolo_box,
    class_distribution,
    load_scene_split,
    prepare_whole_scene_dataset,
)
from src.evaluate import _plot_feature_importance, evaluate_predictions
from src.features import FEATURE_NAMES, N_FEATURES, extract_features
from src.predict import FruitClassifier, checkpoint_available
from src.train import _load_resume_artifacts


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
    assert dist["Apple Unripe"] == 2
    assert dist["Apple Ripe"] == 2
    assert dist["Apple Overripe"] == 1
    assert dist["Mango Overripe"] == 1
    assert len(dist) == len(CLASS_NAMES)


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


def test_resume_rejects_artifact_from_different_dataset_mode(tmp_path):
    (tmp_path / "classifier.joblib").write_bytes(b"")
    (tmp_path / "scaler.joblib").write_bytes(b"")
    (tmp_path / "config.json").write_text(
        json.dumps({"backend": "pytorch", "model_name": "TorchMLP", "dataset_mode": "yolo"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="chế độ dữ liệu khác nhau"):
        _load_resume_artifacts(tmp_path, "cpu", "whole-scene")


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


def test_evaluate_predictions_with_class_names():
    metrics = evaluate_predictions(
        [0, 1], [0, 0],
        class_names=["Apple Unripe", "Apple Ripe"]
    )
    assert "Apple Unripe" in metrics["report"]


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
