"""Kiểm tra upload và gọi inference engine hiện có."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.web.labels import localized_class_name
from src.predict import FruitClassifier, InvalidImageError, ModelNotReadyError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class UploadError(ValueError):
    """Thông tin file upload không hợp lệ."""

    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def safe_filename(filename: str | None) -> str:
    normalized = (filename or "anh-tai-len").replace("\\", "/")
    return Path(normalized).name or "anh-tai-len"


def classifier_supports_probabilities(classifier: Any) -> bool:
    """Chỉ công bố độ tin cậy khi model có predict_proba thực sự."""
    explicit = getattr(classifier, "supports_probability_output", None)
    if explicit is not None:
        return bool(explicit)
    pipeline = getattr(classifier, "pipeline", None)
    if pipeline is not None and hasattr(pipeline, "predict_proba"):
        return True
    model = getattr(classifier, "_classifier", None)
    return callable(getattr(model, "predict_proba", None))


def predict_upload(
    classifier: FruitClassifier,
    *,
    filename: str | None,
    image_bytes: bytes,
    max_upload_bytes: int,
) -> dict[str, Any]:
    """Xác thực ảnh rồi chuyển bytes cho ``FruitClassifier``."""
    clean_name = safe_filename(filename)
    if not image_bytes:
        raise UploadError("empty_file", "Tệp tải lên không có nội dung.")
    if len(image_bytes) > max_upload_bytes:
        raise UploadError("file_too_large", "Ảnh vượt quá giới hạn 10 MB.", 413)
    if Path(clean_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UploadError(
            "unsupported_format", "Định dạng ảnh hỗ trợ: JPG, PNG, BMP và WEBP.", 415
        )
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise UploadError("invalid_image", "Không thể đọc ảnh. Hãy chọn tệp ảnh hợp lệ.") from error

    try:
        result = classifier.predict_from_bytes(image_bytes)
    except (InvalidImageError, OSError, ValueError) as error:
        raise UploadError("invalid_image", str(error)) from error
    except ModelNotReadyError as error:
        raise UploadError("model_not_ready", str(error), 503) from error

    class_name = str(result["class_name"])
    prediction: dict[str, Any] = {
        "filename": clean_name,
        "predicted_class": int(result["class_id"]),
        "class_name": class_name,
        "class_name_vi": localized_class_name(class_name),
        "fruit_type": str(result["fruit"]),
        "ripeness": str(result["ripeness"]),
        "processing_time_ms": float(result.get("processing_time_ms", 0.0)),
    }
    has_probabilities = classifier_supports_probabilities(classifier)
    if has_probabilities and result.get("confidence") is not None:
        prediction["confidence"] = float(result["confidence"])
    probabilities = result.get("probabilities")
    if has_probabilities and probabilities:
        prediction["probabilities"] = {
            str(name): float(value) for name, value in probabilities.items()
        }
    return prediction
