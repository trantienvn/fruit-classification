"""Kiểm toán dữ liệu ảnh phân loại theo thư mục."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.dataset import CLASS_FOLDER_NAMES, IMAGE_EXTENSIONS


def audit_folder_dataset(data_dir: Path) -> dict:
    """Đếm ảnh, kiểm tra giải mã/kích thước và tìm bản sao byte chính xác."""
    report: dict = {
        "data_dir": str(data_dir),
        "splits": {},
        "total_images": 0,
        "corrupt_images": [],
        "duplicate_groups": [],
        "image_dimensions": {},
    }
    hashes: dict[str, list[str]] = defaultdict(list)
    dimensions: Counter[str] = Counter()

    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        class_counts: dict[str, int] = {}
        for folder in sorted(CLASS_FOLDER_NAMES.values()):
            class_dir = split_dir / folder
            image_paths = sorted(
                path for path in class_dir.iterdir()
                if class_dir.is_dir() and path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ) if class_dir.is_dir() else []
            class_counts[folder] = len(image_paths)
            report["total_images"] += len(image_paths)

            for image_path in image_paths:
                try:
                    with Image.open(image_path) as image:
                        dimensions[f"{image.width}x{image.height}"] += 1
                        image.verify()
                    hashes[hashlib.sha256(image_path.read_bytes()).hexdigest()].append(
                        image_path.as_posix()
                    )
                except (OSError, UnidentifiedImageError):
                    report["corrupt_images"].append(image_path.as_posix())
        report["splits"][split] = class_counts

    report["image_dimensions"] = dict(dimensions.most_common())
    report["duplicate_groups"] = [paths for paths in hashes.values() if len(paths) > 1]
    report["cross_split_duplicate_groups"] = [
        paths for paths in report["duplicate_groups"]
        if len({Path(path).parts[-3] for path in paths}) > 1
    ]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng dataset ảnh trái cây.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/dataset"))
    args = parser.parse_args(argv)
    try:
        report = audit_folder_dataset(args.data_dir)
    except OSError as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())