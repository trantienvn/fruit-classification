"""Đọc dataset ảnh nguyên cảnh, YOLO crop và phân loại theo thư mục.

Quy trình:
  - Mặc định train trên ảnh nguyên cảnh đã xuất từ data/raw/
  - Hỗ trợ dataset classification theo thư mục và YOLO crop

Dataset classification dùng nhãn:
  Orange, Banana, Mango × Unripe, Ripe, Spoiled
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Generator

import numpy as np
from PIL import Image, UnidentifiedImageError

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Nhãn của dataset classification hiện tại
CLASS_NAMES = (
    "Orange Unripe",
    "Orange Ripe",
    "Orange Spoiled",
    "Banana Unripe",
    "Banana Ripe",
    "Banana Spoiled",
    "Mango Unripe",
    "Mango Ripe",
    "Mango Spoiled",
)

# Nhãn YOLO cũ giữ riêng vì data/raw vẫn là dataset Apple/Banana/Mango.
LEGACY_CLASS_NAMES = (
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

FRUIT_NAMES = ("Orange", "Banana", "Mango")
RIPENESS_NAMES = ("Unripe", "Ripe", "Spoiled")
CLASS_FOLDER_NAMES = {
    "Orange Unripe": "cam-chua-chin",
    "Orange Ripe": "cam-chin",
    "Orange Spoiled": "cam-hong",
    "Banana Unripe": "chuoi-chua-chin",
    "Banana Ripe": "chuoi-chin",
    "Banana Spoiled": "chuoi-hong",
    "Mango Unripe": "xoai-chua-chin",
    "Mango Ripe": "xoai-chin",
    "Mango Spoiled": "xoai-hong",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Kích thước chuẩn hóa ảnh đầu vào
CROP_SIZE = (128, 128)

# Thư mục dataset
DATA_RAW = Path("data/raw")
DATASET_CLASSIFICATION = Path("data/processed/dataset")
DATASET_WHOLE_SCENE = Path("data/processed/whole-scene")


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
    if not 0 <= class_id < len(LEGACY_CLASS_NAMES):
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


def _parse_yolo_class_id(line: str) -> int | None:
    """Validate a YOLO row and return its class without pixel rounding."""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        cx, cy, width, height = (float(value) for value in parts[1:])
    except ValueError:
        return None
    if not 0 <= class_id < len(LEGACY_CLASS_NAMES):
        return None
    if not np.all(np.isfinite([cx, cy, width, height])):
        return None
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
        return None
    return class_id


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
        label_path = label_dir / f"{img_path.stem}.txt"
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


def load_scene_split(
    data_dir: Path,
    split: str,
    crop_size: tuple[int, int] = CROP_SIZE,
) -> tuple[list[np.ndarray], list[int]]:
    """Nạp ảnh nguyên cảnh từ cấu trúc classification đã chuẩn bị."""
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy split ảnh toàn cảnh: {split_dir}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    for class_id, class_name in enumerate(LEGACY_CLASS_NAMES):
        class_dir = split_dir / class_name.replace(" ", "_")
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Thiếu thư mục lớp trong split: {class_dir}")

        class_count = 0
        image_paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB").resize(crop_size, Image.LANCZOS)
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"Không thể đọc ảnh: {image_path}") from error
            images.append(np.asarray(image, dtype=np.uint8))
            labels.append(class_id)
            class_count += 1

        if class_count == 0:
            raise ValueError(f"Split {split} không có ảnh cho lớp {class_name}.")

    return images, labels


def load_classification_split(
    data_dir: Path,
    split: str,
    crop_size: tuple[int, int] = CROP_SIZE,
) -> tuple[list[np.ndarray], list[int]]:
    """Nạp ảnh theo train/val/test và thư mục lớp của dataset classification."""
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy split dataset: {split_dir}")

    expected_folders = set(CLASS_FOLDER_NAMES.values())
    actual_folders = {
        path.name for path in split_dir.iterdir()
        if path.is_dir()
    }
    missing_folders = expected_folders - actual_folders
    if missing_folders:
        raise FileNotFoundError(
            f"Thiếu thư mục lớp trong {split_dir}: {', '.join(sorted(missing_folders))}"
        )
    unexpected_folders = actual_folders - expected_folders
    if unexpected_folders:
        raise ValueError(
            f"Thư mục lớp không được hỗ trợ trong {split_dir}: "
            f"{', '.join(sorted(unexpected_folders))}"
        )

    images: list[np.ndarray] = []
    labels: list[int] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / CLASS_FOLDER_NAMES[class_name]
        image_paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"Split {split} không có ảnh cho lớp {class_name}.")

        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB").resize(crop_size, Image.LANCZOS)
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"Không thể đọc ảnh: {image_path}") from error
            images.append(np.asarray(image, dtype=np.uint8))
            labels.append(class_id)

    return images, labels


def prepare_whole_scene_dataset(
    source_dir: Path,
    output_dir: Path,
) -> dict:
    """Xuất ảnh nguyên cảnh theo split gốc, chỉ giữ ảnh có duy nhất một class."""
    splits = ("train", "val", "test")
    class_dirs = {name.replace(" ", "_") for name in LEGACY_CLASS_NAMES}
    manifest_path = output_dir / "manifest.json"
    previous_files: list[str] = []

    if manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative_path in previous_manifest.get("files", []):
            rel_path = Path(relative_path)
            if (
                rel_path.is_absolute()
                or len(rel_path.parts) != 3
                or rel_path.parts[0] not in splits
                or rel_path.parts[1] not in class_dirs
            ):
                raise ValueError(f"Đường dẫn file không hợp lệ trong {manifest_path}: {relative_path}")
            previous_files.append(relative_path)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Thư mục đầu ra không rỗng và không có manifest do pipeline tạo: {output_dir}"
        )

    records: list[tuple[Path, Path, int]] = []
    summary = {
        "splits": {},
        "files": [],
    }

    for split in splits:
        image_dir = source_dir / "images" / split
        label_dir = source_dir / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Thiếu thư mục ảnh hoặc nhãn của split {split} trong {source_dir}")

        split_summary = {
            "included": 0,
            "classes": {name: 0 for name in LEGACY_CLASS_NAMES},
            "excluded": {
                "missing_label": 0,
                "empty_label": 0,
                "invalid_label": 0,
                "multiple_classes": 0,
            },
        }
        image_paths = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                split_summary["excluded"]["missing_label"] += 1
                continue

            label_lines = [
                line for line in label_path.read_text(
                    encoding="utf-8-sig", errors="replace"
                ).splitlines() if line.strip()
            ]
            if not label_lines:
                split_summary["excluded"]["empty_label"] += 1
                continue

            class_ids: set[int] = set()
            invalid_label = False
            for line in label_lines:
                class_id = _parse_yolo_class_id(line)
                if class_id is None:
                    invalid_label = True
                    break
                class_ids.add(class_id)

            if invalid_label:
                split_summary["excluded"]["invalid_label"] += 1
                continue
            if len(class_ids) != 1:
                split_summary["excluded"]["multiple_classes"] += 1
                continue

            class_id = class_ids.pop()
            class_name = LEGACY_CLASS_NAMES[class_id]
            relative_path = Path(split) / class_name.replace(" ", "_") / image_path.name
            records.append((image_path, output_dir / relative_path, class_id))
            summary["files"].append(str(relative_path))
            split_summary["included"] += 1
            split_summary["classes"][class_name] += 1

        summary["splits"][split] = split_summary

    output_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in previous_files:
        old_path = output_dir / relative_path
        if old_path.is_file():
            old_path.unlink()
    for source_path, target_path, _ in records:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def class_distribution(
    labels: list[int],
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, int]:
    """Đếm số mẫu theo class."""
    from collections import Counter
    counts = Counter(labels)
    return {class_names[i]: counts.get(i, 0) for i in range(len(class_names))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tạo dataset classification từ ảnh nguyên cảnh có nhãn YOLO."
    )
    parser.add_argument("--source-dir", type=Path, default=DATA_RAW)
    parser.add_argument("--output-dir", type=Path, default=DATASET_WHOLE_SCENE)
    args = parser.parse_args(argv)

    try:
        summary = prepare_whole_scene_dataset(args.source_dir, args.output_dir)
    except (FileExistsError, FileNotFoundError, ValueError, OSError) as error:
        print(f"LỖI: {error}")
        return 2

    print(f"Đã tạo dataset ảnh toàn cảnh tại: {args.output_dir}")
    for split, result in summary["splits"].items():
        excluded = sum(result["excluded"].values())
        print(
            f"  {split}: {result['included']} ảnh được giữ, "
            f"{excluded} ảnh bị loại"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
