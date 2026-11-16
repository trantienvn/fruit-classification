"""
backbone.py — Trình bao bọc CNN backbone (MobileNetV3-Small) để trích xuất đặc trưng.

Thiết kế:
    - Lazy-load: trọng số chỉ được tải khi gọi get_backbone() lần đầu, KHÔNG tải khi import.
    - Frozen: toàn bộ tham số được đóng băng (không huấn luyện lại).
    - Đầu vào: ảnh RGB uint8 shape (H, W, 3) hoặc batch (N, H, W, 3).
    - Đầu ra: vector embedding shape (576,) hoặc (N, 576).
    - CPU-first: không yêu cầu GPU.
    - Xử lý ImportError gracefully khi torch/torchvision chưa được cài.

Hằng số:
    BACKBONE_NAME   = "mobilenetv3_small"
    INPUT_SIZE      = (224, 224)
    EMBEDDING_DIM   = 576
    IMAGENET_MEAN   = [0.485, 0.456, 0.406]
    IMAGENET_STD    = [0.229, 0.224, 0.225]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số kiến trúc
# ---------------------------------------------------------------------------
BACKBONE_NAME: str = "mobilenetv3_small"
INPUT_SIZE: tuple[int, int] = (224, 224)        # (height, width)
EMBEDDING_DIM: int = 576                         # kích thước embedding GAP
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float]  = [0.229, 0.224, 0.225]

# Cache toàn cục để tránh tải lại mô hình nhiều lần
_backbone_instance: Optional["CNNBackbone"] = None

# ---------------------------------------------------------------------------
# Kiểm tra phụ thuộc
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE: bool = False
_TORCHVISION_AVAILABLE: bool = False

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import torchvision  # noqa: F401
    _TORCHVISION_AVAILABLE = True
except ImportError:
    pass

_INSTALL_MSG = (
    "\n[src_dl] Lỗi: torch và/hoặc torchvision chưa được cài đặt.\n"
    "Vui lòng chạy lệnh sau để cài đặt:\n\n"
    "    pip install torch torchvision\n\n"
    "Hoặc dùng uv:\n\n"
    "    uv pip install torch torchvision\n\n"
    "Nếu chỉ cần inference nhẹ hơn, bạn có thể dùng ONNX Runtime thay thế:\n"
    "    pip install onnxruntime\n"
    "    (xem src_dl/backbone_onnx.py nếu có)\n"
)


def _check_dependencies() -> None:
    """Kiểm tra torch + torchvision, raise RuntimeError với hướng dẫn rõ ràng."""
    if not _TORCH_AVAILABLE or not _TORCHVISION_AVAILABLE:
        raise RuntimeError(_INSTALL_MSG)


# ---------------------------------------------------------------------------
# Lớp CNNBackbone
# ---------------------------------------------------------------------------
class CNNBackbone:
    """
    Bộ trích xuất đặc trưng dựa trên MobileNetV3-Small pretrained (ImageNet).

    Chỉ giữ lại phần features + adaptive avg pooling (loại bỏ classifier head).
    Toàn bộ tham số bị đóng băng — không có gradient, không huấn luyện lại.

    Sử dụng:
        backbone = CNNBackbone()
        emb = backbone.extract(img_rgb_uint8)   # shape (576,)
    """

    def __init__(self) -> None:
        _check_dependencies()

        import torch
        import torchvision.models as models

        self.device = torch.device("cpu")

        logger.info("[backbone] Đang tải MobileNetV3-Small pretrained (ImageNet)...")
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        full_model = models.mobilenet_v3_small(weights=weights)

        # Giữ lại features + avgpool, bỏ classifier
        self._features = full_model.features.to(self.device)
        self._avgpool  = full_model.avgpool.to(self.device)

        # Đóng băng toàn bộ tham số
        for param in self._features.parameters():
            param.requires_grad = False
        for param in self._avgpool.parameters():
            param.requires_grad = False

        self._features.eval()
        self._avgpool.eval()

        # Phép chuẩn hóa ImageNet
        self._mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self._std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(1, 1, 3)

        logger.info(
            f"[backbone] Tải xong. Embedding dim = {EMBEDDING_DIM}, device = {self.device}"
        )

    def _preprocess(self, img_rgb_uint8: np.ndarray) -> "torch.Tensor":
        """
        Chuyển ảnh RGB uint8 (H, W, 3) → tensor (1, 3, 224, 224) đã chuẩn hóa ImageNet.

        Tham số:
            img_rgb_uint8: numpy array shape (H, W, 3), dtype uint8, giá trị [0, 255].

        Trả về:
            torch.Tensor shape (1, 3, 224, 224).
        """
        import torch
        from PIL import Image

        if img_rgb_uint8.ndim != 3 or img_rgb_uint8.shape[2] != 3:
            raise ValueError(
                f"Ảnh đầu vào phải có shape (H, W, 3), nhận được {img_rgb_uint8.shape}"
            )

        # Resize về 224×224
        pil = Image.fromarray(img_rgb_uint8.astype(np.uint8), mode="RGB")
        pil = pil.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.BILINEAR)

        # Chuẩn hóa [0,255] → [0,1] rồi ImageNet mean/std
        arr = np.array(pil, dtype=np.float32) / 255.0  # (224, 224, 3)
        arr = (arr - self._mean) / self._std            # (224, 224, 3)
        arr = arr.transpose(2, 0, 1)                    # (3, 224, 224)

        tensor = torch.from_numpy(arr).unsqueeze(0)     # (1, 3, 224, 224)
        return tensor.to(self.device)

    def extract(self, img_rgb_uint8: np.ndarray) -> np.ndarray:
        """
        Trích xuất embedding từ MỘT ảnh.

        Tham số:
            img_rgb_uint8: numpy array shape (H, W, 3), dtype uint8.

        Trả về:
            numpy array shape (576,), dtype float32.
        """
        import torch

        tensor = self._preprocess(img_rgb_uint8)   # (1, 3, 224, 224)
        with torch.no_grad():
            feat = self._features(tensor)          # (1, 576, h', w')
            feat = self._avgpool(feat)             # (1, 576, 1, 1)
            feat = feat.flatten(1)                 # (1, 576)
        return feat.cpu().numpy().squeeze(0)       # (576,)

    def extract_batch(
        self,
        images: list[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Trích xuất embedding cho nhiều ảnh theo batch.

        Tham số:
            images:     danh sách numpy array (H, W, 3), dtype uint8.
            batch_size: số ảnh xử lý mỗi lần (giảm nếu hết RAM).

        Trả về:
            numpy array shape (N, 576), dtype float32.
        """
        import torch

        all_embeddings: list[np.ndarray] = []
        n = len(images)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_imgs = images[start:end]

            tensors = [self._preprocess(img) for img in batch_imgs]  # list of (1,3,224,224)
            batch_tensor = torch.cat(tensors, dim=0)                 # (B, 3, 224, 224)

            with torch.no_grad():
                feat = self._features(batch_tensor)   # (B, 576, h', w')
                feat = self._avgpool(feat)            # (B, 576, 1, 1)
                feat = feat.flatten(1)               # (B, 576)

            all_embeddings.append(feat.cpu().numpy())

            if (start // batch_size) % 10 == 0:
                logger.debug(f"[backbone] Đã xử lý {end}/{n} ảnh")

        return np.vstack(all_embeddings)             # (N, 576)


# ---------------------------------------------------------------------------
# API công khai
# ---------------------------------------------------------------------------

def get_backbone() -> CNNBackbone:
    """
    Lấy (hoặc tạo) singleton CNNBackbone.

    Trọng số được tải lần đầu gọi hàm này, KHÔNG phải khi import module.
    """
    global _backbone_instance
    if _backbone_instance is None:
        _backbone_instance = CNNBackbone()
    return _backbone_instance


def reset_backbone() -> None:
    """Giải phóng singleton (hữu ích khi test hoặc đổi backbone)."""
    global _backbone_instance
    _backbone_instance = None


def extract_embedding(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Tiện ích nhanh: trích xuất embedding từ MỘT ảnh dùng singleton backbone.

    Tham số:
        img_rgb_uint8: numpy array shape (H, W, 3), dtype uint8.

    Trả về:
        numpy array shape (576,), dtype float32.
    """
    return get_backbone().extract(img_rgb_uint8)


def is_available() -> bool:
    """Trả về True nếu torch + torchvision đã được cài đặt."""
    return _TORCH_AVAILABLE and _TORCHVISION_AVAILABLE


if __name__ == "__main__":
    # Kiểm tra nhanh — tạo ảnh ngẫu nhiên và trích xuất embedding
    import sys

    if not is_available():
        print(_INSTALL_MSG)
        sys.exit(1)

    print(f"Backbone: {BACKBONE_NAME}")
    print(f"Embedding dim: {EMBEDDING_DIM}")

    dummy = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    backbone = get_backbone()
    emb = backbone.extract(dummy)
    print(f"Embedding shape: {emb.shape}, dtype: {emb.dtype}")
    print(f"Giá trị: min={emb.min():.4f}, max={emb.max():.4f}, mean={emb.mean():.4f}")
    print("OK — backbone hoạt động bình thường.")
