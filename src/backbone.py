"""
backbone.py — Trình bao bọc CNN backbones để trích xuất đặc trưng tự động.

Hỗ trợ 3 kiến trúc học sâu (Chương 2, Mục 2.4.1):
    1. mobilenetv3_small: Siêu nhẹ, tối ưu Web/Edge, embedding 576-D.
    2. mobilenetv2: Inverted Residuals chuẩn công nghiệp, embedding 1280-D.
    3. resnet18: Mạng Residual sâu kinh điển, trích xuất vân vỏ tốt, embedding 512-D.

Thiết kế:
    - Lazy-load: trọng số chỉ được tải khi gọi get_backbone() lần đầu, KHÔNG tải khi import.
    - Frozen: toàn bộ tham số được đóng băng (không huấn luyện lại).
    - Đầu vào: ảnh RGB uint8 shape (H, W, 3) hoặc batch (N, H, W, 3).
    - Đầu ra: vector embedding shape (D,) hoặc (N, D).
    - Tự động phát hiện và tối ưu trên CUDA GPU (NVIDIA RTX 3050).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cấu hình kiến trúc hỗ trợ
# ---------------------------------------------------------------------------
BACKBONE_SPECS: dict[str, dict[str, any]] = {
    "mobilenetv3_small": {
        "embedding_dim": 576,
        "input_size": (224, 224),
        "display_name": "MobileNetV3-Small",
        "description": "Siêu nhẹ (~4.2 MB ONNX), độ trễ thấp nhất trên trình duyệt",
    },
    "mobilenetv2": {
        "embedding_dim": 1280,
        "input_size": (224, 224),
        "display_name": "MobileNetV2",
        "description": "Chuẩn công nghiệp, Inverted Residuals cân bằng tốc độ và độ chính xác",
    },
    "resnet18": {
        "embedding_dim": 512,
        "input_size": (224, 224),
        "display_name": "ResNet-18",
        "description": "Mạng Residual sâu kinh điển, nhận diện vân bề mặt vỏ quả chi tiết",
    },
}

SUPPORTED_BACKBONES: list[str] = list(BACKBONE_SPECS.keys())
DEFAULT_BACKBONE: str = "mobilenetv3_small"

# Giữ tương thích ngược với các module cũ
BACKBONE_NAME: str = DEFAULT_BACKBONE
INPUT_SIZE: tuple[int, int] = (224, 224)
EMBEDDING_DIM: int = BACKBONE_SPECS[DEFAULT_BACKBONE]["embedding_dim"]
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float]  = [0.229, 0.224, 0.225]

# Cache toàn cục theo (backbone_name, device_str)
_backbone_cache: dict[str, "CNNBackbone"] = {}

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
    "\n[src] Lỗi: torch và/hoặc torchvision chưa được cài đặt.\n"
    "Vui lòng chạy lệnh sau để cài đặt:\n\n"
    "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
)


def _check_dependencies() -> None:
    """Kiểm tra torch + torchvision, raise RuntimeError với hướng dẫn rõ ràng."""
    if not _TORCH_AVAILABLE or not _TORCHVISION_AVAILABLE:
        raise RuntimeError(_INSTALL_MSG)


# ---------------------------------------------------------------------------
# Lớp CNNBackbone Đa Kiến Trúc
# ---------------------------------------------------------------------------
class CNNBackbone:
    """
    Bộ trích xuất đặc trưng đa kiến trúc (MobileNetV3-Small, MobileNetV2, ResNet-18).
    Toàn bộ tham số được đóng băng (frozen) và chỉ dùng chế độ suy luận.
    """

    def __init__(self, name: str = DEFAULT_BACKBONE, device: str | None = None) -> None:
        _check_dependencies()

        import torch
        import torch.nn as nn
        import torchvision.models as models

        norm_name = name.lower().replace("-", "_").strip()
        if norm_name not in BACKBONE_SPECS:
            raise ValueError(
                f"Backbone '{name}' không được hỗ trợ. Lựa chọn hợp lệ: {SUPPORTED_BACKBONES}"
            )

        self.name: str = norm_name
        self.spec: dict = BACKBONE_SPECS[self.name]
        self.embedding_dim: int = self.spec["embedding_dim"]
        self.input_size: tuple[int, int] = self.spec["input_size"]

        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                logger.warning(
                    f"[backbone] Yêu cầu device='cuda' nhưng PyTorch không có CUDA. Chuyển sang CPU."
                )
                self.device = torch.device("cpu")
            else:
                gpu_name = torch.cuda.get_device_name(self.device)
                logger.info(f"[backbone] Sử dụng CUDA GPU: {gpu_name} ({self.device}) cho {self.name}")
        else:
            logger.info(f"[backbone] Sử dụng CPU ({self.device}) cho {self.name}")

        logger.info(f"[backbone] Đang tải {self.spec['display_name']} pretrained (ImageNet)...")

        # Tải weights tương ứng theo từng kiến trúc
        if self.name == "mobilenetv3_small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            full_model = models.mobilenet_v3_small(weights=weights)
            self._features = full_model.features.to(self.device)
            self._pool = full_model.avgpool.to(self.device)
        elif self.name == "mobilenetv2":
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            full_model = models.mobilenet_v2(weights=weights)
            self._features = full_model.features.to(self.device)
            self._pool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        elif self.name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            full_model = models.resnet18(weights=weights)
            # ResNet18: conv1 -> bn1 -> relu -> maxpool -> layer1..4
            self._features = nn.Sequential(
                full_model.conv1,
                full_model.bn1,
                full_model.relu,
                full_model.maxpool,
                full_model.layer1,
                full_model.layer2,
                full_model.layer3,
                full_model.layer4,
            ).to(self.device)
            self._pool = full_model.avgpool.to(self.device)
        else:
            raise NotImplementedError(f"Chưa hỗ trợ backbone {self.name}")

        # Đóng băng toàn bộ tham số
        for param in self._features.parameters():
            param.requires_grad = False
        for param in self._pool.parameters():
            param.requires_grad = False

        self._features.eval()
        self._pool.eval()

        # Phép chuẩn hóa ImageNet
        self._mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self._std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(1, 1, 3)

        logger.info(
            f"[backbone] Tải xong {self.name}. Embedding dim = {self.embedding_dim}, device = {self.device}"
        )

    def _preprocess(self, img_rgb_uint8: np.ndarray) -> "torch.Tensor":
        """Chuyển ảnh RGB uint8 (H, W, 3) → tensor (1, 3, 224, 224) chuẩn hóa ImageNet."""
        import torch
        from PIL import Image

        if img_rgb_uint8.ndim != 3 or img_rgb_uint8.shape[2] != 3:
            raise ValueError(
                f"Ảnh đầu vào phải có shape (H, W, 3), nhận được {img_rgb_uint8.shape}"
            )

        pil = Image.fromarray(img_rgb_uint8.astype(np.uint8), mode="RGB")
        pil = pil.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)

        arr = np.array(pil, dtype=np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        arr = arr.transpose(2, 0, 1)

        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor.to(self.device)

    def extract(self, img_rgb_uint8: np.ndarray) -> np.ndarray:
        """Trích xuất embedding từ MỘT ảnh."""
        import torch

        tensor = self._preprocess(img_rgb_uint8)
        use_autocast = self.device.type == "cuda"
        with torch.inference_mode():
            if use_autocast:
                with torch.amp.autocast(device_type="cuda"):
                    feat = self._features(tensor)
                    feat = self._pool(feat)
            else:
                feat = self._features(tensor)
                feat = self._pool(feat)
            feat = feat.flatten(1)
        return feat.cpu().numpy().squeeze(0).astype(np.float32)

    def extract_batch(
        self,
        images: list[np.ndarray],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Trích xuất embedding cho nhiều ảnh theo batch trên CUDA GPU."""
        import torch

        if batch_size is None:
            batch_size = 64 if self.device.type == "cuda" else 32

        all_embeddings: list[np.ndarray] = []
        n = len(images)
        use_autocast = self.device.type == "cuda"

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_imgs = images[start:end]

            tensors = [self._preprocess(img) for img in batch_imgs]
            batch_tensor = torch.cat(tensors, dim=0)

            with torch.inference_mode():
                if use_autocast:
                    with torch.amp.autocast(device_type="cuda"):
                        feat = self._features(batch_tensor)
                        feat = self._pool(feat)
                else:
                    feat = self._features(batch_tensor)
                    feat = self._pool(feat)
                feat = feat.flatten(1)

            all_embeddings.append(feat.cpu().numpy().astype(np.float32))

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return np.vstack(all_embeddings)

    def get_torch_feature_extractor(self) -> "torch.nn.Module":
        """
        Trả về một nn.Sequential gộp _features + _pool + Flatten
        để dùng cho export ONNX hoặc phân tích kiến trúc / tính FLOPs.
        """
        import torch.nn as nn
        return nn.Sequential(
            self._features,
            self._pool,
            nn.Flatten(1),
        )


# ---------------------------------------------------------------------------
# API công khai
# ---------------------------------------------------------------------------

def get_backbone(name: str = DEFAULT_BACKBONE, device: str | None = None) -> CNNBackbone:
    """Lấy (hoặc tạo) instance CNNBackbone theo tên kiến trúc và thiết bị."""
    global _backbone_cache
    norm_name = name.lower().replace("-", "_").strip()
    cache_key = f"{norm_name}:{device or 'auto'}"

    if cache_key not in _backbone_cache:
        _backbone_cache[cache_key] = CNNBackbone(name=norm_name, device=device)
    return _backbone_cache[cache_key]


def reset_backbone() -> None:
    """Giải phóng cache backbone và bộ nhớ CUDA."""
    global _backbone_cache
    for inst in _backbone_cache.values():
        if inst.device.type == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    _backbone_cache.clear()


def extract_embedding(
    img_rgb_uint8: np.ndarray,
    name: str = DEFAULT_BACKBONE,
    device: str | None = None,
) -> np.ndarray:
    """Tiện ích trích xuất embedding từ 1 ảnh với kiến trúc chỉ định."""
    return get_backbone(name=name, device=device).extract(img_rgb_uint8)


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
