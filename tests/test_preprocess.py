"""Kiểm thử chuyển đổi ảnh sang dữ liệu số (.npz) và tích hợp train/predict."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.dataset import CLASS_FOLDER_NAMES, CLASS_NAMES, LEGACY_CLASS_NAMES
from src.predict import FruitClassifier, ModelNotReadyError
from src.preprocess import (
    DEFAULT_CACHE_FILE,
    MANIFEST_FILE,
    N_PIXEL_FEATURES,
    PIXEL_IMAGE_SIZE,
    REPRESENTATION_NAME,
    _compute_dataset_hash,
    cache_is_valid,
    image_to_numeric_vector,
    load_preprocessed,
    numeric_vectors_to_images,
    preprocess,
)
from src.train import train


def _make_fake_folder_dataset(data_dir: Path, n_per_class: int = 1) -> None:
    """Tạo dataset giả cấu trúc Cam/Chuối/Xoài với ảnh không trùng giữa các split."""
    split_offset = {"train": 0, "val": 80, "test": 160}
    for split in ("train", "val", "test"):
        for idx, class_name in enumerate(CLASS_NAMES):
            class_dir = data_dir / split / CLASS_FOLDER_NAMES[class_name]
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                r = (idx * 25 + split_offset[split] + i * 7) % 256
                g = (40 + idx * 17 + i * 31 + split_offset[split]) % 256
                b = (90 + i * 19 + split_offset[split] // 2) % 256
                img = Image.new("RGB", (48, 36), (r, g, b))
                img.save(class_dir / f"sample_{i}.png")


def _make_fake_whole_scene_dataset(data_dir: Path, n_per_class: int = 1) -> None:
    """Tạo dataset giả cấu trúc whole-scene."""
    split_offset = {"train": 0, "val": 90, "test": 180}
    for split in ("train", "val", "test"):
        for idx, class_name in enumerate(LEGACY_CLASS_NAMES):
            class_dir = data_dir / split / class_name.replace(" ", "_")
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                r = (idx * 23 + split_offset[split]) % 256
                g = (60 + i * 37 + split_offset[split]) % 256
                b = (110 + split_offset[split] // 2) % 256
                img = Image.new("RGB", (40, 40), (r, g, b))
                img.save(class_dir / f"sample_{i}.png")


def test_image_to_numeric_vector_resizes_and_normalizes():
    """Chuyển ảnh RGB bất kỳ kích thước thành vector float32 [0, 1] 64×64×3."""
    img = np.full((32, 48, 3), 255, dtype=np.uint8)
    vec = image_to_numeric_vector(img)
    assert vec.shape == (N_PIXEL_FEATURES,)
    assert vec.dtype == np.float32
    np.testing.assert_allclose(vec, 1.0, atol=1e-6)

    zero_img = np.zeros((64, 64, 3), dtype=np.uint8)
    zero_vec = image_to_numeric_vector(zero_img)
    np.testing.assert_array_equal(zero_vec, np.zeros(N_PIXEL_FEATURES, dtype=np.float32))


def test_numeric_vectors_to_images_roundtrip():
    """Khôi phục ảnh uint8 từ ma trận số chuẩn hóa [0, 1] giữ nguyên giá trị pixel."""
    rng = np.random.default_rng(42)
    imgs = [rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8) for _ in range(3)]
    X = np.vstack([image_to_numeric_vector(img) for img in imgs])
    restored = numeric_vectors_to_images(X, image_size=PIXEL_IMAGE_SIZE)

    assert len(restored) == 3
    for orig, rec in zip(imgs, restored):
        assert rec.dtype == np.uint8
        assert rec.shape == (64, 64, 3)
        np.testing.assert_array_equal(orig, rec)


def test_preprocess_creates_numeric_cache_for_folder_dataset(tmp_path):
    """Tiền xử lý tạo file .npz và manifest.json cho dataset Cam/Chuối/Xoài."""
    data_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _make_fake_folder_dataset(data_dir, n_per_class=1)

    result_path = preprocess(data_dir=data_dir, cache_dir=cache_dir)

    assert result_path.is_file()
    assert (cache_dir / DEFAULT_CACHE_FILE).is_file()
    assert (cache_dir / MANIFEST_FILE).is_file()

    data = load_preprocessed(cache_dir=cache_dir)
    assert data["X_train"].shape == (len(CLASS_NAMES), N_PIXEL_FEATURES)
    assert data["X_val"].shape == (len(CLASS_NAMES), N_PIXEL_FEATURES)
    assert data["X_test"].shape == (len(CLASS_NAMES), N_PIXEL_FEATURES)
    assert data["X_train"].dtype == np.float32
    assert 0.0 <= float(data["X_train"].min()) <= float(data["X_train"].max()) <= 1.0
    assert list(data["class_names"]) == list(CLASS_NAMES)
    assert tuple(int(v) for v in data["image_size"]) == PIXEL_IMAGE_SIZE
    assert str(data["representation"][0]) == REPRESENTATION_NAME
    assert str(data["dataset_mode"][0]) == "folder"


def test_preprocess_supports_whole_scene_dataset(tmp_path):
    """Tiền xử lý tự nhận diện cả cấu trúc whole-scene."""
    data_dir = tmp_path / "whole-scene"
    cache_dir = tmp_path / "cache"
    _make_fake_whole_scene_dataset(data_dir, n_per_class=1)

    preprocess(data_dir=data_dir, cache_dir=cache_dir)
    data = load_preprocessed(cache_dir=cache_dir)

    assert data["X_train"].shape == (len(LEGACY_CLASS_NAMES), N_PIXEL_FEATURES)
    assert list(data["class_names"]) == list(LEGACY_CLASS_NAMES)
    assert str(data["dataset_mode"][0]) == "whole-scene"


def test_preprocess_skips_when_cache_valid_and_force_reprocesses(tmp_path, capsys):
    """Bỏ qua khi cache hợp lệ và xử lý lại khi có force=True."""
    data_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _make_fake_folder_dataset(data_dir, n_per_class=1)

    preprocess(data_dir=data_dir, cache_dir=cache_dir)
    capsys.readouterr()

    preprocess(data_dir=data_dir, cache_dir=cache_dir)
    assert "Cache dữ liệu số hợp lệ" in capsys.readouterr().out

    preprocess(data_dir=data_dir, cache_dir=cache_dir, force=True)
    assert "TIỀN XỬ LÝ" in capsys.readouterr().out


def test_cache_invalidated_when_dataset_or_size_changes(tmp_path):
    """Cache bị vô hiệu khi thêm ảnh mới hoặc đổi kích thước ảnh."""
    data_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _make_fake_folder_dataset(data_dir, n_per_class=1)

    assert not cache_is_valid(data_dir=data_dir, cache_dir=cache_dir)
    preprocess(data_dir=data_dir, cache_dir=cache_dir)
    assert cache_is_valid(data_dir=data_dir, cache_dir=cache_dir)
    assert not cache_is_valid(data_dir=data_dir, cache_dir=cache_dir, image_size=(32, 32))

    extra_img = Image.new("RGB", (48, 36), (255, 0, 0))
    extra_img.save(data_dir / "train" / CLASS_FOLDER_NAMES[CLASS_NAMES[0]] / "extra.png")
    assert not cache_is_valid(data_dir=data_dir, cache_dir=cache_dir)


def test_load_preprocessed_raises_when_missing(tmp_path):
    """Ném FileNotFoundError khi chưa có dữ liệu số."""
    with pytest.raises(FileNotFoundError, match="Chưa có dữ liệu số"):
        load_preprocessed(cache_dir=tmp_path)


def test_dataset_hash_changes_with_new_file(tmp_path):
    """Hash thay đổi khi thêm ảnh mới."""
    data_dir = tmp_path / "dataset"
    _make_fake_folder_dataset(data_dir, n_per_class=1)
    hash1 = _compute_dataset_hash(data_dir)

    Image.new("RGB", (32, 32), (0, 0, 255)).save(
        data_dir / "train" / CLASS_FOLDER_NAMES[CLASS_NAMES[0]] / "new_image.png"
    )
    hash2 = _compute_dataset_hash(data_dir)
    assert hash1 != hash2


def test_manifest_contains_metadata(tmp_path):
    """Manifest lưu metadata đầy đủ của dữ liệu số."""
    data_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _make_fake_folder_dataset(data_dir, n_per_class=1)

    preprocess(data_dir=data_dir, cache_dir=cache_dir)
    manifest = json.loads((cache_dir / MANIFEST_FILE).read_text(encoding="utf-8"))

    assert manifest["representation"] == REPRESENTATION_NAME
    assert manifest["image_size"] == [64, 64]
    assert manifest["n_features"] == N_PIXEL_FEATURES
    assert manifest["dataset_mode"] == "folder"
    assert manifest["class_names"] == list(CLASS_NAMES)


def test_train_and_predict_use_numeric_pixels_without_handcrafted_features(
    tmp_path, monkeypatch
):
    """Train dùng bộ dữ liệu số .npz, không gọi trích xuất đặc trưng thủ công."""
    def fail_handcrafted(*args, **kwargs):
        raise AssertionError("Không được gọi trích xuất đặc trưng thủ công!")

    monkeypatch.setattr("src.features.extract_features", fail_handcrafted)
    monkeypatch.setattr("src.features.build_feature_matrix", fail_handcrafted)
    monkeypatch.setattr(
        "src.train.MODELS",
        {
            "Dummy": DummyClassifier(strategy="most_frequent"),
            "LogisticRegression": LogisticRegression(max_iter=200, random_state=42),
        },
    )

    data_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "numeric"
    model_dir = tmp_path / "models"
    _make_fake_folder_dataset(data_dir, n_per_class=6)

    # Lần train 1: tự tiền xử lý -> tạo .npz -> train bằng dữ liệu số
    metrics1 = train(
        data_dir=data_dir,
        model_dir=model_dir,
        use_pca=True,
        run_gridsearch=False,
        cache_dir=cache_dir,
    )
    assert 0.0 <= metrics1["accuracy"] <= 1.0
    assert (cache_dir / DEFAULT_CACHE_FILE).is_file()

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assert config["feature_type"] == REPRESENTATION_NAME
    assert config["n_features"] == N_PIXEL_FEATURES
    assert config["crop_size"] == [64, 64]
    assert config["dataset_mode"] == "folder"
    assert config["class_names"] == list(CLASS_NAMES)

    # Lần train 2: dùng lại cache .npz, không chạy preprocess lại
    def fail_preprocess(*args, **kwargs):
        raise AssertionError("Cache hợp lệ thì train phải nạp .npz thay vì chạy preprocess lại!")

    monkeypatch.setattr("src.train.preprocess", fail_preprocess)
    metrics2 = train(
        data_dir=data_dir,
        model_dir=model_dir,
        use_pca=True,
        run_gridsearch=False,
        cache_dir=cache_dir,
    )
    assert metrics2["accuracy"] == metrics1["accuracy"]

    # Dự đoán ảnh mới bằng FruitClassifier dùng cùng chuẩn pixel số
    clf = FruitClassifier(model_dir).load()
    result = clf.predict(Image.new("RGB", (100, 80), (40, 120, 90)))
    assert result["class_name"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_rejects_legacy_handcrafted_config(tmp_path):
    """FruitClassifier báo lỗi rõ ràng nếu nạp model đặc trưng thủ công cũ."""
    (tmp_path / "classifier.joblib").write_bytes(b"placeholder")
    (tmp_path / "scaler.joblib").write_bytes(b"placeholder")
    (tmp_path / "config.json").write_text(
        json.dumps({"class_names": list(CLASS_NAMES), "n_features": 90, "crop_size": [128, 128]}),
        encoding="utf-8",
    )

    with pytest.raises(ModelNotReadyError, match="python -m src.train"):
        FruitClassifier(tmp_path).load()
