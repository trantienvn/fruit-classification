"""Kiểm thử các module ML pipeline — không cần dataset thật."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.dataset import CLASS_NAMES, _parse_yolo_box, class_distribution
from src.evaluate import evaluate_predictions
from src.features import N_FEATURES, extract_features
from src.predict import FruitClassifier, checkpoint_available


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
