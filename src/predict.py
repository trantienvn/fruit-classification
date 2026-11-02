"""Pipeline inference — nhận ảnh, trả kết quả dự đoán.

Sử dụng:
  from src.predict import FruitClassifier

  clf = FruitClassifier()               # nạp model từ models/
  result = clf.predict_from_file("anh.jpg")
  # result = {
  #   "class_id": 1,
  #   "class_name": "Orange Ripe",
  #   "fruit": "Orange",
  #   "ripeness": "Ripe",
  #   "confidence": 0.87,
  #   "probabilities": {"Orange Unripe": 0.05, "Orange Ripe": 0.87, ...},
  #   "processing_time_ms": 45.2,
  # }
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.dataset import CLASS_NAMES, CROP_SIZE, FRUIT_NAMES, RIPENESS_NAMES
from src.features import extract_features

# Thư mục model mặc định
DEFAULT_MODEL_DIR = Path("models")

# Tên file trong model_dir
CLASSIFIER_FILE = "classifier.joblib"
SCALER_FILE = "scaler.joblib"
PCA_FILE = "pca.joblib"
CONFIG_FILE = "config.json"

# Kích thước ảnh tối đa cho phép upload (10 MB)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Định dạng ảnh hỗ trợ
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ModelNotReadyError(Exception):
    """Model chưa được huấn luyện hoặc chưa được lưu."""


class InvalidImageError(Exception):
    """Ảnh đầu vào không hợp lệ."""


class FruitClassifier:
    """Pipeline inference phân loại trái cây.

    Nạp một lần, dự đoán nhiều lần — không fit lại preprocessing.
    """

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._classifier = None
        self._scaler = None
        self._pca = None
        self._config: dict[str, Any] = {}
        self._class_names = list(CLASS_NAMES)
        self._loaded = False

    # ── Nạp model ─────────────────────────────────────────────────────────────

    def load(self) -> "FruitClassifier":
        """Nạp model, scaler và config từ model_dir. Gọi một lần."""
        model_dir = self.model_dir
        clf_path = model_dir / CLASSIFIER_FILE
        scaler_path = model_dir / SCALER_FILE
        config_path = model_dir / CONFIG_FILE

        for p in (clf_path, scaler_path, config_path):
            if not p.is_file():
                raise ModelNotReadyError(
                    f"Thiếu file model: {p}\n"
                    "Hãy chạy: python -m src.train để huấn luyện model trước."
                )

        self._config = json.loads(config_path.read_text(encoding="utf-8"))
        class_names = self._config.get("class_names", list(CLASS_NAMES))
        if (
            not isinstance(class_names, list)
            or not class_names
            or not all(isinstance(name, str) for name in class_names)
        ):
            raise ModelNotReadyError(f"Danh sách class_names trong {config_path} không hợp lệ.")
        if self._config.get("n_classes", len(class_names)) != len(class_names):
            raise ModelNotReadyError(f"Số lớp trong {config_path} không khớp class_names.")
        self._class_names = class_names
        self._classifier = joblib.load(clf_path)
        self._scaler = joblib.load(scaler_path)

        pca_path = model_dir / PCA_FILE
        if self._config.get("use_pca") and pca_path.is_file():
            self._pca = joblib.load(pca_path)

        self._loaded = True
        return self

    @property
    def is_ready(self) -> bool:
        """Kiểm tra model đã được nạp."""
        return self._loaded

    @classmethod
    def model_exists(cls, model_dir: Path | str = DEFAULT_MODEL_DIR) -> bool:
        """Kiểm tra các file model cần thiết đã tồn tại."""
        md = Path(model_dir)
        return all((md / f).is_file() for f in (CLASSIFIER_FILE, SCALER_FILE, CONFIG_FILE))

    def get_info(self) -> dict[str, Any]:
        """Trả về thông tin model đang dùng."""
        if not self._loaded:
            return {"status": "Chưa nạp model"}
        return {
            "model_name": self._config.get("model_name", "unknown"),
            "n_classes": self._config.get("n_classes", 9),
            "class_names": self._class_names,
            "n_features": self._config.get("n_features", 0),
            "use_pca": self._config.get("use_pca", False),
            "val_metrics": self._config.get("val_metrics", {}),
            "test_metrics": self._config.get("test_metrics", {}),
            "n_train": self._config.get("n_train", "N/A"),
            "status": "Sẵn sàng",
        }

    # ── Predict ───────────────────────────────────────────────────────────────

    def _preprocess_image(self, img: Image.Image) -> np.ndarray:
        """Chuyển PIL Image → ndarray uint8 RGB (crop_size)."""
        img_rgb = img.convert("RGB")
        crop_size = tuple(self._config.get("crop_size", CROP_SIZE))
        img_resized = img_rgb.resize(crop_size, Image.LANCZOS)  # type: ignore[arg-type]
        return np.array(img_resized, dtype=np.uint8)

    def _run_inference(self, arr: np.ndarray) -> dict[str, Any]:
        """Chạy inference trên ảnh đã tiền xử lý."""
        feat = extract_features(arr).reshape(1, -1)
        feat_scaled = self._scaler.transform(feat)  # type: ignore[union-attr]
        if self._pca is not None:
            feat_scaled = self._pca.transform(feat_scaled)

        class_id = int(self._classifier.predict(feat_scaled)[0])  # type: ignore[union-attr]
        if not 0 <= class_id < len(self._class_names):
            raise ModelNotReadyError(
                f"Model dự đoán class_id={class_id}, ngoài phạm vi cấu hình."
            )
        class_name = self._class_names[class_id]

        # Phân tách tên class thành fruit + ripeness
        parts = class_name.split()
        fruit = parts[0]
        ripeness = parts[1] if len(parts) > 1 else "Unknown"

        # Confidence và probabilities
        probs: dict[str, float] = {}
        confidence = 1.0
        if hasattr(self._classifier, "predict_proba"):
            prob_arr = self._classifier.predict_proba(feat_scaled)[0]  # type: ignore[union-attr]
            classes = self._class_names
            if len(prob_arr) != len(classes):
                raise ModelNotReadyError("Số xác suất model trả về không khớp cấu hình lớp.")
            probs = {classes[i]: float(p) for i, p in enumerate(prob_arr)}
            confidence = float(prob_arr[class_id])
        elif hasattr(self._classifier, "decision_function"):
            scores = self._classifier.decision_function(feat_scaled)[0]  # type: ignore[union-attr]
            exp_scores = np.exp(scores - np.max(scores))
            prob_arr = exp_scores / exp_scores.sum()
            classes = self._class_names
            if len(prob_arr) != len(classes):
                raise ModelNotReadyError("Số điểm model trả về không khớp cấu hình lớp.")
            probs = {classes[i]: float(p) for i, p in enumerate(prob_arr)}
            confidence = float(prob_arr[class_id])

        return {
            "class_id": class_id,
            "class_name": class_name,
            "fruit": fruit,
            "ripeness": ripeness,
            "confidence": confidence,
            "probabilities": probs,
        }

    def predict(self, pil_image: Image.Image) -> dict[str, Any]:
        """Dự đoán từ PIL Image. Gọi load() trước.

        Trả về dict kết quả dự đoán với processing_time_ms.
        """
        if not self._loaded:
            raise ModelNotReadyError("Hãy gọi .load() trước.")

        t0 = time.perf_counter()
        arr = self._preprocess_image(pil_image)
        result = self._run_inference(arr)
        result["processing_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result

    def predict_from_path(self, image_path: Path | str) -> dict[str, Any]:
        """Dự đoán từ đường dẫn file ảnh."""
        path = Path(image_path)
        if not path.is_file():
            raise InvalidImageError(f"Không tìm thấy file: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise InvalidImageError(f"Định dạng không hỗ trợ: {path.suffix}")
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            raise InvalidImageError("File vượt quá 10 MB.")
        try:
            img = Image.open(path)
        except (OSError, UnidentifiedImageError) as e:
            raise InvalidImageError(f"Không thể đọc ảnh: {e}") from e
        return self.predict(img)

    def predict_from_bytes(self, data: bytes) -> dict[str, Any]:
        """Dự đoán từ bytes (từ Streamlit uploader)."""
        import io
        if len(data) > MAX_FILE_SIZE_BYTES:
            raise InvalidImageError("File vượt quá 10 MB.")
        try:
            img = Image.open(io.BytesIO(data))
        except (OSError, UnidentifiedImageError) as e:
            raise InvalidImageError(f"Không thể đọc ảnh: {e}") from e
        return self.predict(img)


# Singleton — nạp một lần, dùng nhiều lần trong Streamlit
_classifier_instance: FruitClassifier | None = None


def get_classifier(model_dir: Path | str = DEFAULT_MODEL_DIR) -> FruitClassifier:
    """Trả về FruitClassifier đã nạp (singleton)."""
    global _classifier_instance
    if _classifier_instance is None or not _classifier_instance.is_ready:
        _classifier_instance = FruitClassifier(model_dir)
        _classifier_instance.load()
    return _classifier_instance


def checkpoint_available(path: str | Path = DEFAULT_MODEL_DIR) -> bool:
    """Kiểm tra model đã sẵn sàng (tương thích ngược với code cũ)."""
    return FruitClassifier.model_exists(path)
