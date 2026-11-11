"""Chuyển đổi bộ ảnh thành dữ liệu dạng số (.npz) từ pixel RGB chuẩn hóa [0, 1].

Thay vì trích xuất 90 đặc trưng thủ công, mô-đun này resize từng ảnh về
kích thước cố định (mặc định 64×64 RGB), chuẩn hóa giá trị pixel về miền
[0.0, 1.0], làm phẳng thành vector số 12.288 chiều và lưu vào file .npz.

Mỗi lần huấn luyện, `src.train` sẽ nạp trực tiếp bộ dữ liệu số đã tiền xử lý
này mà không cần đọc lại ảnh gốc hay trích xuất đặc trưng thủ công.

Cách dùng:

    # Bước 1: Chuyển đổi bộ ảnh sang dữ liệu số (chỉ chạy 1 lần hoặc khi đổi dữ liệu)
    python -m src.preprocess

    # Bước 2: Huấn luyện (tự động nạp dữ liệu số từ cache)
    python -m src.train

Cấu trúc file .npz lưu:
    X_train, y_train  — ma trận pixel chuẩn hóa [0, 1] float32 (n, 12288) và nhãn
    X_val,   y_val    — ma trận pixel chuẩn hóa [0, 1] float32 (n, 12288) và nhãn
    X_test,  y_test   — ma trận pixel chuẩn hóa [0, 1] float32 (n, 12288) và nhãn
    class_names       — danh sách tên lớp tương ứng với dataset
    image_size        — kích thước [width, height], mặc định [64, 64]
    representation    — định danh biểu diễn số ("pixels_rgb_0_1")
    dataset_mode      — "folder" hoặc "whole-scene"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from PIL import Image

from src.dataset import (
    CLASS_FOLDER_NAMES,
    CLASS_NAMES,
    DATASET_CLASSIFICATION,
    LEGACY_CLASS_NAMES,
    deduplicate_splits,
    load_classification_split,
    load_scene_split,
)

# Cấu hình chuyển đổi ảnh -> vector số mặc định
PIXEL_IMAGE_SIZE: tuple[int, int] = (64, 64)  # (width, height)
PIXEL_CHANNELS = 3
N_PIXEL_FEATURES = PIXEL_IMAGE_SIZE[0] * PIXEL_IMAGE_SIZE[1] * PIXEL_CHANNELS
REPRESENTATION_NAME = "pixels_rgb_0_1"

# Đường dẫn mặc định
DEFAULT_DATA_DIR = DATASET_CLASSIFICATION
DEFAULT_CACHE_DIR = Path("data/processed/numeric")
DEFAULT_CACHE_FILE = "numeric_dataset.npz"
MANIFEST_FILE = "manifest.json"

SplitLoader = Callable[[Path, str, tuple[int, int]], tuple[list[np.ndarray], list[int]]]


def detect_dataset_format(data_dir: Path) -> tuple[str, tuple[str, ...], SplitLoader]:
    """Tự nhận diện cấu trúc dataset (folder Cam/Chuối/Xoài hoặc whole-scene)."""
    train_dir = data_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục split train trong: {data_dir}")

    subfolders = {path.name for path in train_dir.iterdir() if path.is_dir()}
    folder_expected = set(CLASS_FOLDER_NAMES.values())
    scene_expected = {name.replace(" ", "_") for name in LEGACY_CLASS_NAMES}

    if folder_expected <= subfolders:
        return "folder", CLASS_NAMES, load_classification_split
    if scene_expected <= subfolders:
        return "whole-scene", LEGACY_CLASS_NAMES, load_scene_split

    raise ValueError(
        f"Cấu trúc thư mục trong {train_dir} không khớp dataset classification "
        "hoặc whole-scene."
    )


def pixel_feature_names(image_size: tuple[int, int] = PIXEL_IMAGE_SIZE) -> list[str]:
    """Sinh danh sách tên tọa độ pixel (px_y{y}_x{x}_{c}) cho biểu đồ báo cáo."""
    width, height = image_size
    return [
        f"px_y{y}_x{x}_{channel}"
        for y in range(height)
        for x in range(width)
        for channel in ("r", "g", "b")
    ]


def image_to_numeric_vector(
    img_rgb: np.ndarray,
    image_size: tuple[int, int] = PIXEL_IMAGE_SIZE,
) -> np.ndarray:
    """Chuyển ảnh RGB uint8 (H×W×3) thành vector số float32 chuẩn hóa [0, 1].

    Dùng nhất quán ở bước tiền xử lý, huấn luyện (augmentation) và dự đoán.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != PIXEL_CHANNELS:
        raise ValueError("Cần ảnh RGB 3 kênh với shape (H, W, 3).")

    width, height = image_size
    if img_rgb.shape[0] != height or img_rgb.shape[1] != width:
        resized = Image.fromarray(img_rgb).resize((width, height), Image.LANCZOS)
        arr = np.asarray(resized, dtype=np.float32)
    else:
        arr = img_rgb.astype(np.float32, copy=False)

    return (arr / np.float32(255.0)).reshape(-1)


def images_to_numeric_matrix(
    images: list[np.ndarray],
    image_size: tuple[int, int] = PIXEL_IMAGE_SIZE,
) -> np.ndarray:
    """Chuyển danh sách ảnh RGB thành ma trận số float32 shape (n_samples, W*H*3)."""
    n_dims = image_size[0] * image_size[1] * PIXEL_CHANNELS
    if not images:
        return np.empty((0, n_dims), dtype=np.float32)
    return np.vstack(
        [image_to_numeric_vector(img, image_size=image_size) for img in images]
    ).astype(np.float32, copy=False)


def numeric_vectors_to_images(
    X: np.ndarray,
    image_size: tuple[int, int] = PIXEL_IMAGE_SIZE,
) -> list[np.ndarray]:
    """Khôi phục danh sách ảnh uint8 (H, W, 3) từ ma trận số chuẩn hóa [0, 1]."""
    width, height = image_size
    expected_dims = width * height * PIXEL_CHANNELS
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != expected_dims:
        raise ValueError(
            f"Ma trận số phải có shape (n, {expected_dims}), nhận được {arr.shape}."
        )
    pixels = np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    reshaped = pixels.reshape(-1, height, width, PIXEL_CHANNELS)
    return [reshaped[i] for i in range(len(reshaped))]


def augment_training_images(
    images: list[np.ndarray],
    labels: list[int] | np.ndarray,
    random_state: int = 42,
) -> tuple[list[np.ndarray], list[int]]:
    """Tạo một ảnh xoay/lật/đổi sáng cho mỗi mẫu train; không nhận val/test."""
    rng = np.random.default_rng(random_state)
    augmented_images = list(images)
    augmented_labels = [int(label) for label in labels]
    for image, label in zip(images, labels):
        height, width = image.shape[:2]
        angle = float(rng.uniform(-15, 15))
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        variant = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        if rng.random() < 0.5:
            variant = cv2.flip(variant, 1)
        factor = float(rng.uniform(0.85, 1.15))
        offset = float(rng.uniform(-12, 12))
        variant = np.clip(variant.astype(np.float32) * factor + offset, 0, 255).astype(
            np.uint8
        )
        augmented_images.append(variant)
        augmented_labels.append(int(label))
    return augmented_images, augmented_labels


def _compute_dataset_hash(data_dir: Path) -> str:
    """Tính hash của dataset dựa trên đường dẫn tương đối, kích thước và mtime."""
    hasher = hashlib.sha256()
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.rglob("*")):
            if path.is_file():
                hasher.update(path.relative_to(data_dir).as_posix().encode("utf-8"))
                stat = path.stat()
                hasher.update(str(stat.st_size).encode("utf-8"))
                hasher.update(str(int(stat.st_mtime)).encode("utf-8"))
    return hasher.hexdigest()[:16]


def preprocess(
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
    image_size: tuple[int, int] = PIXEL_IMAGE_SIZE,
    force: bool = False,
) -> Path:
    """Chuyển đổi bộ dữ liệu ảnh sang ma trận số chuẩn hóa [0, 1] và lưu .npz."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size phải có chiều rộng và chiều cao dương.")
    normalized_size = (width, height)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / DEFAULT_CACHE_FILE
    manifest_path = cache_dir / MANIFEST_FILE

    dataset_hash = _compute_dataset_hash(data_dir)
    if not force and cache_is_valid(
        data_dir=data_dir, cache_dir=cache_dir, image_size=normalized_size
    ):
        print(f"Cache dữ liệu số hợp lệ, bỏ qua tiền xử lý: {cache_path}")
        return cache_path

    dataset_mode, class_names, split_loader = detect_dataset_format(data_dir)

    print("=" * 60)
    print("TIỀN XỬ LÝ — CHUYỂN ĐỔI ẢNH SANG DỮ LIỆU SỐ (PIXEL RGB [0, 1])")
    print(f"Dataset   : {data_dir} ({dataset_mode})")
    print(f"Kích thước: {width}x{height}x{PIXEL_CHANNELS} = {width * height * PIXEL_CHANNELS} chiều")
    print(f"Lưu tại   : {cache_path}")
    print("=" * 60)

    print("\nĐang nạp tập train...")
    t0 = time.perf_counter()
    train_imgs, y_train = split_loader(data_dir, "train", normalized_size)
    print(f"  {len(y_train)} mẫu — {time.perf_counter() - t0:.1f}s")

    print("Đang nạp tập val...")
    t0 = time.perf_counter()
    val_imgs, y_val = split_loader(data_dir, "val", normalized_size)
    print(f"  {len(y_val)} mẫu — {time.perf_counter() - t0:.1f}s")

    print("Đang nạp tập test...")
    t0 = time.perf_counter()
    test_imgs, y_test = split_loader(data_dir, "test", normalized_size)
    print(f"  {len(y_test)} mẫu — {time.perf_counter() - t0:.1f}s")

    (train_imgs, y_train), (val_imgs, y_val), (test_imgs, y_test), duplicates = (
        deduplicate_splits(train_imgs, y_train, val_imgs, y_val, test_imgs, y_test)
    )
    dropped = sum(duplicates.values())
    if dropped:
        print(f"  Loại {dropped} ảnh trùng chính xác (test > val > train): {duplicates}")

    print("\nChuyển đổi ảnh sang ma trận số chuẩn hóa [0, 1]...")
    t0 = time.perf_counter()
    X_train = images_to_numeric_matrix(train_imgs, image_size=normalized_size)
    X_val = images_to_numeric_matrix(val_imgs, image_size=normalized_size)
    X_test = images_to_numeric_matrix(test_imgs, image_size=normalized_size)
    print(f"  Xong — {time.perf_counter() - t0:.1f}s")
    print(f"  Train : {X_train.shape} ({X_train.dtype})")
    print(f"  Val   : {X_val.shape} ({X_val.dtype})")
    print(f"  Test  : {X_test.shape} ({X_test.dtype})")

    print(f"\nĐang lưu dữ liệu số vào {cache_path}...")
    t0 = time.perf_counter()
    np.savez_compressed(
        cache_path,
        X_train=X_train,
        y_train=np.asarray(y_train, dtype=np.int64),
        X_val=X_val,
        y_val=np.asarray(y_val, dtype=np.int64),
        X_test=X_test,
        y_test=np.asarray(y_test, dtype=np.int64),
        class_names=np.asarray(list(class_names)),
        image_size=np.asarray([width, height], dtype=np.int64),
        representation=np.asarray([REPRESENTATION_NAME]),
        dataset_mode=np.asarray([dataset_mode]),
    )
    file_size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"  Xong — {time.perf_counter() - t0:.1f}s | {file_size_mb:.2f} MB")

    n_features = int(X_train.shape[1]) if X_train.ndim == 2 else width * height * PIXEL_CHANNELS
    manifest = {
        "dataset_hash": dataset_hash,
        "data_dir": str(data_dir),
        "dataset_mode": dataset_mode,
        "representation": REPRESENTATION_NAME,
        "image_size": [width, height],
        "n_features": n_features,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "class_names": list(class_names),
        "duplicates_dropped": duplicates,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nTiền xử lý hoàn tất. Chạy huấn luyện bằng:")
    print("  python -m src.train")
    return cache_path


def load_preprocessed(cache_dir: Path | None = None) -> dict[str, Any]:
    """Nạp dữ liệu số đã tiền xử lý từ file .npz (kèm metadata manifest nếu có)."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_path = cache_dir / DEFAULT_CACHE_FILE
    manifest_path = cache_dir / MANIFEST_FILE
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Chưa có dữ liệu số đã tiền xử lý tại {cache_path}.\n"
            "Hãy chạy: python -m src.preprocess"
        )

    print(f"Nạp dữ liệu số từ {cache_path}...")
    t0 = time.perf_counter()
    with np.load(cache_path, allow_pickle=False) as archive:
        data: dict[str, Any] = {key: archive[key] for key in archive.files}

    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            data["duplicates_dropped"] = manifest.get(
                "duplicates_dropped", {"train": 0, "val": 0, "test": 0}
            )
        except (json.JSONDecodeError, OSError):
            data["duplicates_dropped"] = {"train": 0, "val": 0, "test": 0}
    else:
        data["duplicates_dropped"] = {"train": 0, "val": 0, "test": 0}

    print(
        f"  Xong — {time.perf_counter() - t0:.2f}s | "
        f"train={data['X_train'].shape}, "
        f"val={data['X_val'].shape}, "
        f"test={data['X_test'].shape}"
    )
    return data


def cache_is_valid(
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
    image_size: tuple[int, int] = PIXEL_IMAGE_SIZE,
) -> bool:
    """Kiểm tra file .npz đã tồn tại và khớp dataset, kích thước ảnh, biểu diễn số."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_path = cache_dir / DEFAULT_CACHE_FILE
    manifest_path = cache_dir / MANIFEST_FILE

    if not cache_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (
            manifest.get("dataset_hash") == _compute_dataset_hash(data_dir)
            and manifest.get("representation") == REPRESENTATION_NAME
            and manifest.get("image_size") == [int(image_size[0]), int(image_size[1])]
        )
    except (json.JSONDecodeError, KeyError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chuyển đổi bộ ảnh sang dữ liệu số chuẩn hóa [0, 1] (.npz)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Thư mục chứa dataset ảnh (mặc định: data/processed/dataset)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Thư mục lưu file .npz (mặc định: data/processed/numeric)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=PIXEL_IMAGE_SIZE[0],
        help="Kích thước cạnh ảnh vuông (mặc định: 64 -> 64x64x3 = 12288 chiều)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Xử lý lại ngay cả khi cache đã hợp lệ",
    )
    args = parser.parse_args(argv)

    try:
        preprocess(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            image_size=(args.size, args.size),
            force=args.force,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"LỖI: {exc}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
