"""
predict_dl.py — Inference cho mô hình DL (CNN embedding + sklearn classifier).

Lớp FruitClassifierDL hoàn toàn độc lập với src/predict.py.
Không import từ src/ (ngoại trừ src_dl/).

Pipeline dự đoán:
    PIL Image (bất kỳ kích thước)
        → numpy RGB uint8
        → CNN backbone (MobileNetV3-Small)
        → embedding (576,)
        → StandardScaler (+ PCA nếu được bật)
        → sklearn classifier
        → dict kết quả

Chạy trực tiếp (CLI):
    python -m src_dl.predict_dl <đường_dẫn_ảnh> [--model-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số mặc định
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR: Path = Path("models_dl")

# ---------------------------------------------------------------------------
# Ngoại lệ riêng (độc lập, không import từ src/)
# ---------------------------------------------------------------------------

class ModelNotReadyError(RuntimeError):
    """Mô hình chưa được tải hoặc không tìm thấy."""
    pass


class InvalidImageError(ValueError):
    """Ảnh đầu vào không hợp lệ hoặc không đọc được."""
    pass


# ---------------------------------------------------------------------------
# Lớp FruitClassifierDL
# ---------------------------------------------------------------------------

class FruitClassifierDL:
    """
    Bộ phân loại trái cây dựa trên CNN embedding + sklearn classifier.

    Độc lập hoàn toàn với FruitClassifier trong src/predict.py.

    Thuộc tính sau khi load():
        model_dir       : Path đến thư mục mô hình
        class_names     : list[str] — tên các lớp
        feature_type    : str — phải bắt đầu bằng "cnn_"
        backbone_name   : str — tên backbone (vd: "mobilenetv3_small")
        embedding_dim   : int — chiều embedding (vd: 576)
        pipeline        : sklearn Pipeline đã tải (scaler + pca + clf)

    Sử dụng:
        clf = FruitClassifierDL()
        clf.load()
        result = clf.predict_from_path("ảnh.jpg")
        print(result["label"], result["confidence"])
    """

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, device: str = "auto") -> None:
        self.model_dir:     Path          = Path(model_dir)
        self.device:        str           = device
        self.class_names:   list[str]     = []
        self.feature_type:  str           = ""
        self.backbone_name: str           = ""
        self.embedding_dim: int           = 0
        self.pipeline:      Optional[Any] = None
        self._loaded:       bool          = False
        self._backbone      = None        # lazy-loaded CNNBackbone

    # ------------------------------------------------------------------
    # Tải mô hình
    # ------------------------------------------------------------------

    def load(self, model_dir: Optional[Path] = None) -> "FruitClassifierDL":
        """
        Tải pipeline sklearn và cấu hình từ model_dir.

        Tham số:
            model_dir: ghi đè thư mục mô hình (tùy chọn).

        Trả về:
            self (hỗ trợ chaining: clf.load().predict(...))

        Raise:
            ModelNotReadyError nếu file không tồn tại hoặc feature_type không hợp lệ.
        """
        try:
            import joblib
        except ImportError:
            from sklearn.externals import joblib  # type: ignore

        if model_dir is not None:
            self.model_dir = Path(model_dir)

        clf_path    = self.model_dir / "classifier_dl.joblib"
        config_path = self.model_dir / "config.json"

        # Kiểm tra file tồn tại
        if not clf_path.exists():
            raise ModelNotReadyError(
                f"Không tìm thấy mô hình: {clf_path}\n"
                "Hãy huấn luyện trước:\n\n"
                "    python -m src_dl.train_dl\n"
            )
        if not config_path.exists():
            raise ModelNotReadyError(
                f"Không tìm thấy config: {config_path}"
            )

        # Đọc config
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        feature_type = config.get("feature_type", "")
        if not feature_type.startswith("cnn_"):
            raise ModelNotReadyError(
                f"feature_type không hợp lệ: '{feature_type}'. "
                "Mô hình DL phải có feature_type bắt đầu bằng 'cnn_'.\n"
                "Nếu muốn dùng mô hình ML truyền thống, hãy dùng src/predict.py."
            )

        self.feature_type  = feature_type
        self.backbone_name = config.get("backbone", "mobilenetv3_small")
        self.embedding_dim = config.get("embedding_dim", 576)
        self.class_names   = config.get("class_names", [])

        # Tải pipeline
        self.pipeline = joblib.load(clf_path)
        self._loaded  = True

        logger.info(
            f"[predict_dl] Tải xong mô hình '{self.backbone_name}' "
            f"| {len(self.class_names)} lớp | feature_type={self.feature_type}"
        )
        return self

    def _ensure_loaded(self) -> None:
        if not self._loaded or self.pipeline is None:
            raise ModelNotReadyError(
                "Mô hình chưa được tải. Hãy gọi clf.load() trước."
            )

    # ------------------------------------------------------------------
    # Lấy backbone (lazy)
    # ------------------------------------------------------------------

    def _get_backbone(self):
        if self._backbone is None:
            from src_dl.backbone import get_backbone
            self._backbone = get_backbone(device=self.device)
        return self._backbone

    # ------------------------------------------------------------------
    # Dự đoán
    # ------------------------------------------------------------------

    def predict(self, pil_image) -> dict:
        """
        Dự đoán lớp từ PIL Image.

        Tham số:
            pil_image: PIL.Image.Image (bất kỳ mode/kích thước).

        Trả về:
            dict với các khóa:
                label       : str  — tên lớp dự đoán
                label_index : int  — chỉ số lớp
                confidence  : float — xác suất cao nhất [0, 1]
                probabilities : dict[str, float] — xác suất tất cả lớp
                feature_type : str
                backbone     : str
        """
        self._ensure_loaded()

        from PIL import Image
        if not isinstance(pil_image, Image.Image):
            raise InvalidImageError(
                f"Đầu vào phải là PIL.Image.Image, nhận được {type(pil_image)}"
            )

        # Chuyển sang RGB nếu cần
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        img_arr = np.array(pil_image, dtype=np.uint8)

        # Trích xuất CNN embedding
        try:
            backbone = self._get_backbone()
            embedding = backbone.extract(img_arr)       # (576,)
        except RuntimeError as e:
            raise InvalidImageError(f"Lỗi trích xuất embedding: {e}") from e

        # Dự đoán qua sklearn pipeline
        X = embedding.reshape(1, -1)                   # (1, 576)
        label_idx = int(self.pipeline.predict(X)[0])

        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(X)[0]  # (n_classes,)
        else:
            # Mô hình không hỗ trợ predict_proba → one-hot
            probs = np.zeros(len(self.class_names), dtype=np.float32)
            probs[label_idx] = 1.0

        confidence = float(probs[label_idx])
        label = (
            self.class_names[label_idx]
            if 0 <= label_idx < len(self.class_names)
            else str(label_idx)
        )

        prob_dict = {
            cls: round(float(p), 6)
            for cls, p in zip(self.class_names, probs)
        }

        return {
            "label":         label,
            "label_index":   label_idx,
            "confidence":    round(confidence, 6),
            "probabilities": prob_dict,
            "feature_type":  self.feature_type,
            "backbone":      self.backbone_name,
        }

    def predict_from_path(self, image_path: str | Path) -> dict:
        """
        Dự đoán từ đường dẫn file ảnh.

        Tham số:
            image_path: đường dẫn đến file ảnh (str hoặc Path).

        Trả về:
            Như predict().
        """
        from PIL import Image, UnidentifiedImageError

        path = Path(image_path)
        if not path.exists():
            raise InvalidImageError(f"File không tồn tại: {path}")

        try:
            pil = Image.open(path)
        except UnidentifiedImageError as e:
            raise InvalidImageError(f"Không thể mở ảnh {path}: {e}") from e

        return self.predict(pil)

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        """
        Dự đoán từ dữ liệu ảnh dạng bytes (vd: upload từ web).

        Tham số:
            image_bytes: dữ liệu ảnh thô (JPEG/PNG/...).

        Trả về:
            Như predict().
        """
        import io
        from PIL import Image, UnidentifiedImageError

        try:
            pil = Image.open(io.BytesIO(image_bytes))
        except (UnidentifiedImageError, Exception) as e:
            raise InvalidImageError(f"Không thể đọc ảnh từ bytes: {e}") from e

        return self.predict(pil)

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"FruitClassifierDL("
            f"backbone={self.backbone_name!r}, "
            f"classes={len(self.class_names)}, "
            f"status={status})"
        )


# ---------------------------------------------------------------------------
# Tiện ích nhanh
# ---------------------------------------------------------------------------

_default_classifier: Optional[FruitClassifierDL] = None


def get_classifier(model_dir: Path = DEFAULT_MODEL_DIR) -> FruitClassifierDL:
    """Lấy (hoặc tạo) singleton FruitClassifierDL."""
    global _default_classifier
    if _default_classifier is None or not _default_classifier._loaded:
        _default_classifier = FruitClassifierDL(model_dir).load()
    return _default_classifier


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dự đoán lớp trái cây từ ảnh (CNN embedding + sklearn)"
    )
    p.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Đường dẫn đến file ảnh cần dự đoán",
    )
    p.add_argument(
        "--model-dir", type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Thư mục chứa mô hình (mặc định: {DEFAULT_MODEL_DIR})",
    )
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Thiết bị tính toán (mặc định: auto - dùng CUDA nếu có)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Bật logging chi tiết",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level  = logging.DEBUG if args.verbose else logging.WARNING,
        format = "%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.image is None:
        parser.print_help()
        raise SystemExit(0)

    try:
        clf = FruitClassifierDL(args.model_dir, device=args.device)
        clf.load()
        result = clf.predict_from_path(args.image)

        print(f"\nKết quả dự đoán cho: {args.image}")
        print(f"  Lớp:        {result['label']}")
        print(f"  Độ tin cậy: {result['confidence']:.2%}")
        print(f"  Backbone:   {result['backbone']}")
        print("\nXác suất tất cả lớp:")
        for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"  {cls:<30s} {prob:.4f}  {bar}")

    except ModelNotReadyError as e:
        print(f"\n✗ Mô hình chưa sẵn sàng:\n{e}")
        raise SystemExit(1)
    except InvalidImageError as e:
        print(f"\n✗ Ảnh không hợp lệ:\n{e}")
        raise SystemExit(1)
    except RuntimeError as e:
        # Lỗi thiếu torch/torchvision
        print(f"\n✗ Lỗi runtime:\n{e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
