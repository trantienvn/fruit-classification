"""Đọc dataset YOLO và tạo samples cho phân loại ảnh.

Quy trình:
  - Đọc cặp ảnh/nhãn YOLO từ data/raw/
  - Crop ảnh theo bounding box
  - Mỗi crop là một sample với nhãn class (0–8)

Nhãn class:
  0: Apple Unripe   1: Apple Ripe   2: Apple Overripe
  3: Banana Unripe  4: Banana Ripe  5: Banana Overripe
  6: Mango Unripe   7: Mango Ripe   8: Mango Overripe
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
from PIL import Image, UnidentifiedImageError

# Nhãn 9 class
CLASS_NAMES = (
    "Apple Unripe",
    "Apple Ripe",
    "Apple Overripe",
    "Banana Unripe",
    "Banana Ripe",
    "Banana Overripe",
    "Mango Unripe",
    "Mango Ripe",
    "Mango Overripe",
)

# Tên loại trái cây và độ chín
FRUIT_NAMES = ("Apple", "Banana", "Mango")
RIPENESS_NAMES = ("Unripe", "Ripe", "Overripe")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Kích thước resize cho mỗi crop
CROP_SIZE = (128, 128)

# Thư mục dataset mặc định
DATA_RAW = Path("data/raw")


def _parse_yolo_box(line: str, img_w: int, img_h: int) -> tuple[int, int, int, int, int] | None:
    """Chuyển dòng YOLO sang (class_id, x1, y1, x2, y2).

    Trả về None nếu dòng không hợp lệ.
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        cx, cy, bw, bh = (float(x) for x in parts[1:])
    except ValueError:
        return None
    if not (0 <= class_id <= 8):
        return None

    x1 = int(round((cx - bw / 2) * img_w))
    y1 = int(round((cy - bh / 2) * img_h))
    x2 = int(round((cx + bw / 2) * img_w))
    y2 = int(round((cy + bh / 2) * img_h))

    # Clamp về trong ảnh
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    if x2 <= x1 or y2 <= y1:
        return None
    return class_id, x1, y1, x2, y2


def iter_samples(
    data_dir: Path = DATA_RAW,
    split: str = "train",
    crop_size: tuple[int, int] = CROP_SIZE,
) -> Generator[tuple[np.ndarray, int], None, None]:
    """Sinh từng cặp (crop_array, class_id) từ dataset YOLO.

    crop_array có shape (H, W, 3), dtype uint8.
    """
    image_dir = data_dir / "images" / split
    label_dir = data_dir / "labels" / split

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục nhãn: {label_dir}")

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for img_path in image_paths:
        label_path = (label_dir / img_path.stem).with_suffix(".txt")
        if not label_path.is_file():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except (OSError, UnidentifiedImageError):
            continue

        img_w, img_h = img.size
        label_text = label_path.read_text(encoding="utf-8-sig", errors="replace")

        for line in label_text.splitlines():
            if not line.strip():
                continue
            parsed = _parse_yolo_box(line, img_w, img_h)
            if parsed is None:
                continue
            class_id, x1, y1, x2, y2 = parsed

            crop = img.crop((x1, y1, x2, y2))
            crop = crop.resize(crop_size, Image.LANCZOS)
            yield np.array(crop, dtype=np.uint8), class_id


def load_split(
    data_dir: Path = DATA_RAW,
    split: str = "train",
    crop_size: tuple[int, int] = CROP_SIZE,
) -> tuple[list[np.ndarray], list[int]]:
    """Nạp toàn bộ split vào bộ nhớ.

    Trả về (images, labels):
      images — danh sách ndarray uint8 shape (H, W, 3)
      labels — danh sách int class_id
    """
    images: list[np.ndarray] = []
    labels: list[int] = []
    for img_arr, class_id in iter_samples(data_dir, split, crop_size):
        images.append(img_arr)
        labels.append(class_id)
    return images, labels


def class_distribution(labels: list[int]) -> dict[str, int]:
    """Đếm số mẫu theo class."""
    from collections import Counter
    counts = Counter(labels)
    return {CLASS_NAMES[i]: counts.get(i, 0) for i in range(len(CLASS_NAMES))}

