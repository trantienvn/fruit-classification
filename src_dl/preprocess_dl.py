"""
preprocess_dl.py — Tiền xử lý ảnh và trích xuất CNN embedding (MobileNetV3-Small).

Pipeline:
    Ảnh gốc (H×W×3 uint8)
        → [tăng cường dữ liệu trên ảnh — chỉ train]
        → Resize 224×224
        → Chuẩn hóa ImageNet
        → CNN backbone (frozen)
        → Embedding (576,)
        → Lưu cache .npz

Lưu ý quan trọng:
    Tăng cường (augmentation) phải thực hiện trên ẢNH GỐC trước khi trích xuất CNN.
    Khác với src/preprocess.py — không thể tái tạo ảnh từ embedding CNN.

Hằng số:
    DEFAULT_CACHE_DIR  = data/processed/dl_features/mobilenetv3_small
    REPRESENTATION_NAME = "cnn_mobilenetv3_small"
    CACHE_FILE         = dl_dataset.npz

Chạy trực tiếp:
    python -m src_dl.preprocess_dl [--data-dir PATH] [--cache-dir PATH] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR: Path  = Path("data/processed/dataset")
DEFAULT_CACHE_DIR: Path = Path("data/processed/dl_features/mobilenetv3_small")
CACHE_FILE: str         = "dl_dataset.npz"
MANIFEST_FILE: str      = "manifest.json"
REPRESENTATION_NAME: str = "cnn_mobilenetv3_small"

# Tham số tăng cường (chỉ áp dụng cho tập train)
AUG_ROTATE_DEG: float = 15.0    # xoay ngẫu nhiên ±15°
AUG_FLIP_PROB:  float = 0.5     # lật ngang 50% xác suất
AUG_BRIGHT_RANGE: tuple[float, float] = (0.85, 1.15)  # độ sáng ±15%
AUG_N_COPIES: int = 2           # số bản tăng cường thêm cho mỗi ảnh train


# ---------------------------------------------------------------------------
# Hàm tăng cường ảnh (trên PIL Image)
# ---------------------------------------------------------------------------

def _augment_image(pil_img: Image.Image, rng: random.Random) -> Image.Image:
    """
    Áp dụng ngẫu nhiên: xoay, lật ngang, thay đổi độ sáng.

    Tham số:
        pil_img: ảnh PIL gốc.
        rng:     bộ sinh số ngẫu nhiên (để tái lập).

    Trả về:
        Ảnh PIL đã biến đổi.
    """
    # Xoay ngẫu nhiên ±AUG_ROTATE_DEG độ
    angle = rng.uniform(-AUG_ROTATE_DEG, AUG_ROTATE_DEG)
    img = pil_img.rotate(angle, resample=Image.BILINEAR, expand=False)

    # Lật ngang
    if rng.random() < AUG_FLIP_PROB:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Thay đổi độ sáng
    factor = rng.uniform(*AUG_BRIGHT_RANGE)
    img = ImageEnhance.Brightness(img).enhance(factor)

    return img


def _load_split_images(
    data_dir: Path,
    split: str,                       # "train" | "val" | "test"
    class_names: list[str],
) -> tuple[list[np.ndarray], list[int]]:
    """
    Tải ảnh từ thư mục theo cấu trúc:
        data_dir / split / class_name / *.jpg|*.png|*.jpeg|*.webp

    Trả về:
        images: danh sách numpy array (H, W, 3) uint8.
        labels: danh sách nhãn (int).
    """
    images: list[np.ndarray] = []
    labels: list[int]        = []
    split_dir = data_dir / split

    if not split_dir.exists():
        logger.warning(f"[preprocess_dl] Thư mục {split_dir} không tồn tại, bỏ qua.")
        return images, labels

    for idx, cls in enumerate(class_names):
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            logger.warning(f"[preprocess_dl] Lớp '{cls}' không tìm thấy trong {split_dir}")
            continue

        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        img_paths = sorted(
            p for p in cls_dir.iterdir() if p.suffix.lower() in exts
        )

        for path in img_paths:
            try:
                pil = Image.open(path).convert("RGB")
                images.append(np.array(pil, dtype=np.uint8))
                labels.append(idx)
            except Exception as exc:
                logger.warning(f"[preprocess_dl] Bỏ qua ảnh lỗi {path}: {exc}")

    logger.info(f"[preprocess_dl] Split '{split}': tải {len(images)} ảnh, {len(set(labels))} lớp")
    return images, labels


def _augment_train_set(
    images: list[np.ndarray],
    labels: list[int],
    n_copies: int = AUG_N_COPIES,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Tạo thêm n_copies bản tăng cường cho mỗi ảnh train.

    Trả về danh sách kết hợp ảnh gốc + ảnh tăng cường.
    """
    rng = random.Random(seed)
    aug_images: list[np.ndarray] = list(images)
    aug_labels: list[int]        = list(labels)

    for img, lbl in zip(images, labels):
        pil = Image.fromarray(img, mode="RGB")
        for _ in range(n_copies):
            aug = _augment_image(pil, rng)
            aug_images.append(np.array(aug, dtype=np.uint8))
            aug_labels.append(lbl)

    logger.info(
        f"[preprocess_dl] Tăng cường: {len(images)} → {len(aug_images)} ảnh train"
    )
    return aug_images, aug_labels


# ---------------------------------------------------------------------------
# Tạo manifest (hash)
# ---------------------------------------------------------------------------

def _compute_manifest(data_dir: Path, backbone_name: str) -> dict:
    """Tạo manifest dict mô tả cache (dùng để kiểm tra tính hợp lệ)."""
    h = hashlib.md5(str(data_dir.resolve()).encode()).hexdigest()[:8]
    return {
        "backbone": backbone_name,
        "representation": REPRESENTATION_NAME,
        "data_dir_hash": h,
        "aug_copies": AUG_N_COPIES,
        "aug_rotate_deg": AUG_ROTATE_DEG,
        "aug_flip_prob": AUG_FLIP_PROB,
    }


def cache_is_valid(cache_dir: Path, data_dir: Path, backbone_name: str) -> bool:
    """
    Kiểm tra xem cache .npz có hợp lệ không (tồn tại + manifest khớp).

    Trả về True nếu có thể dùng lại cache.
    """
    npz_path      = cache_dir / CACHE_FILE
    manifest_path = cache_dir / MANIFEST_FILE

    if not npz_path.exists() or not manifest_path.exists():
        return False

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        expected = _compute_manifest(data_dir, backbone_name)
        return saved == expected
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Hàm chính: trích xuất và lưu cache
# ---------------------------------------------------------------------------

def extract_and_cache(
    data_dir: Path      = DEFAULT_DATA_DIR,
    cache_dir: Path     = DEFAULT_CACHE_DIR,
    backbone_name: str  = "mobilenetv3_small",
    force: bool         = False,
    batch_size: int     = 32,
    seed: int           = 42,
    dataset_mode: str   = "dataset",
) -> Path:
    """
    Trích xuất CNN embedding cho toàn bộ dataset và lưu vào .npz.

    Tham số:
        data_dir:      thư mục chứa train/val/test.
        cache_dir:     nơi lưu dl_dataset.npz.
        backbone_name: hiện chỉ hỗ trợ "mobilenetv3_small".
        force:         bỏ qua cache cũ, trích xuất lại.
        batch_size:    kích thước batch cho backbone.
        seed:          hạt giống ngẫu nhiên cho augmentation.
        dataset_mode:  "dataset" | "whole-scene" (ghi vào manifest).

    Trả về:
        Path đến file .npz đã lưu.
    """
    from src_dl.backbone import get_backbone, EMBEDDING_DIM  # lazy import

    cache_dir = Path(cache_dir)
    data_dir  = Path(data_dir)
    npz_path  = cache_dir / CACHE_FILE

    # Kiểm tra cache
    if not force and cache_is_valid(cache_dir, data_dir, backbone_name):
        logger.info(f"[preprocess_dl] Cache hợp lệ tại {npz_path}, bỏ qua trích xuất.")
        return npz_path

    logger.info("[preprocess_dl] Bắt đầu trích xuất CNN embedding...")
    logger.info(f"  data_dir  = {data_dir}")
    logger.info(f"  cache_dir = {cache_dir}")
    logger.info(f"  backbone  = {backbone_name}")

    # Lấy danh sách lớp từ thư mục train
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục train: {train_dir}\n"
            "Hãy đảm bảo dataset đã được giải nén vào data/processed/dataset/"
        )

    class_names = sorted(
        d.name for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not class_names:
        raise ValueError(f"Không tìm thấy lớp nào trong {train_dir}")

    logger.info(f"[preprocess_dl] Lớp ({len(class_names)}): {class_names}")

    # Khởi tạo backbone (tải trọng số lần đầu)
    backbone = get_backbone()

    result: dict[str, np.ndarray | list] = {}

    for split in ["train", "val", "test"]:
        images, labels = _load_split_images(data_dir, split, class_names)

        if not images:
            # Tạo mảng rỗng để tránh lỗi khi lưu .npz
            result[f"X_{split}"] = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
            result[f"y_{split}"] = np.empty((0,), dtype=np.int32)
            continue

        # Chỉ tăng cường tập train
        if split == "train":
            images, labels = _augment_train_set(images, labels, seed=seed)

        logger.info(f"[preprocess_dl] Trích xuất embedding cho '{split}' ({len(images)} ảnh)...")
        embeddings = backbone.extract_batch(images, batch_size=batch_size)

        result[f"X_{split}"] = embeddings.astype(np.float32)
        result[f"y_{split}"] = np.array(labels, dtype=np.int32)

        logger.info(
            f"[preprocess_dl] Split '{split}': X shape = {result[f'X_{split}'].shape}"
        )

    # Lưu .npz
    cache_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        npz_path,
        X_train     = result["X_train"],
        y_train     = result["y_train"],
        X_val       = result.get("X_val",  np.empty((0, EMBEDDING_DIM), dtype=np.float32)),
        y_val       = result.get("y_val",  np.empty((0,), dtype=np.int32)),
        X_test      = result.get("X_test", np.empty((0, EMBEDDING_DIM), dtype=np.float32)),
        y_test      = result.get("y_test", np.empty((0,), dtype=np.int32)),
        class_names = np.array(class_names),
    )

    # Lưu manifest
    manifest = _compute_manifest(data_dir, backbone_name)
    manifest.update({
        "embedding_dim": EMBEDDING_DIM,
        "dataset_mode":  dataset_mode,
        "n_train":       int(result["X_train"].shape[0]),
        "n_val":         int(result.get("X_val",  np.empty((0,))).shape[0]),
        "n_test":        int(result.get("X_test", np.empty((0,))).shape[0]),
        "n_classes":     len(class_names),
        "class_names":   class_names,
    })

    with open(cache_dir / MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"[preprocess_dl] Đã lưu cache tại {npz_path}")
    return npz_path


def load_cached_features(cache_dir: Path = DEFAULT_CACHE_DIR) -> dict:
    """
    Tải CNN embedding từ cache .npz.

    Trả về dict với các khóa:
        X_train, y_train, X_val, y_val, X_test, y_test, class_names
    """
    npz_path = Path(cache_dir) / CACHE_FILE
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy cache tại {npz_path}.\n"
            "Hãy chạy trước:\n\n"
            "    python -m src_dl.preprocess_dl\n"
        )

    data = np.load(npz_path, allow_pickle=True)
    return {
        "X_train":     data["X_train"],
        "y_train":     data["y_train"],
        "X_val":       data["X_val"],
        "y_val":       data["y_val"],
        "X_test":      data["X_test"],
        "y_test":      data["y_test"],
        "class_names": list(data["class_names"]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trích xuất CNN embedding (MobileNetV3-Small) và lưu cache .npz"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Thư mục dataset (mặc định: {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Nơi lưu cache (mặc định: {DEFAULT_CACHE_DIR})",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv3_small",
        choices=["mobilenetv3_small"],
        help="Backbone CNN (mặc định: mobilenetv3_small)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size cho backbone (mặc định: 32)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Hạt giống ngẫu nhiên (mặc định: 42)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Bỏ qua cache cũ, trích xuất lại từ đầu",
    )
    p.add_argument(
        "--dataset-mode",
        type=str,
        default="dataset",
        choices=["dataset", "whole-scene"],
        help="Chế độ dataset (mặc định: dataset)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Bật logging chi tiết (DEBUG)",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        npz_path = extract_and_cache(
            data_dir     = args.data_dir,
            cache_dir    = args.cache_dir,
            backbone_name= args.backbone,
            force        = args.force,
            batch_size   = args.batch_size,
            seed         = args.seed,
            dataset_mode = args.dataset_mode,
        )
        print(f"\n✓ Trích xuất hoàn tất. Cache lưu tại:\n  {npz_path.resolve()}")
    except RuntimeError as e:
        # Lỗi thiếu torch/torchvision
        print(str(e))
        raise SystemExit(1)
    except FileNotFoundError as e:
        print(f"\n✗ Lỗi: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
