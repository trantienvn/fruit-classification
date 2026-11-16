"""Kiểm tra upload và thực thi suy luận mô hình học sâu (FruitClassifierDL)."""

from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.web.labels import get_fruit_analysis, localized_class_name

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class UploadError(ValueError):
    """Thông tin file upload không hợp lệ."""

    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class InvalidImageError(ValueError):
    """Ảnh đầu vào không hợp lệ."""
    pass


class ModelNotReadyError(RuntimeError):
    """Mô hình chưa sẵn sàng."""
    pass


def safe_filename(filename: str | None) -> str:
    normalized = (filename or "anh-tai-len").replace("\\", "/")
    return Path(normalized).name or "anh-tai-len"


def classifier_supports_probabilities(classifier: Any) -> bool:
    """Kiểm tra xem classifier có cung cấp xác suất dự đoán hay không."""
    explicit = getattr(classifier, "supports_probability_output", None)
    if explicit is not None:
        return bool(explicit)
    pipeline = getattr(classifier, "pipeline", None)
    if pipeline is not None and hasattr(pipeline, "predict_proba"):
        return True
    model = getattr(classifier, "_classifier", None)
    return callable(getattr(model, "predict_proba", None))


def predict_upload(
    classifier: Any,
    *,
    filename: str | None,
    image_bytes: bytes,
    max_upload_bytes: int,
) -> dict[str, Any]:
    """Xác thực ảnh rồi chuyển bytes cho mô hình Deep Learning phân loại."""
    clean_name = safe_filename(filename)
    if not image_bytes:
        raise UploadError("empty_file", "Tệp tải lên không có nội dung.")
    if len(image_bytes) > max_upload_bytes:
        raise UploadError("file_too_large", "Ảnh vượt quá giới hạn 10 MB.", 413)

    suffix = Path(clean_name).suffix.lower()
    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        raise UploadError(
            "unsupported_format", "Định dạng ảnh hỗ trợ: JPG, PNG, BMP và WEBP.", 415
        )

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img.verify()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise UploadError("invalid_image", "Không thể đọc ảnh. Hãy chọn tệp ảnh hợp lệ.") from error

    t0 = time.perf_counter()
    try:
        result = classifier.predict_from_bytes(image_bytes)
    except Exception as error:
        if "not ready" in str(error).lower() or isinstance(error, ModelNotReadyError):
            raise UploadError("model_not_ready", str(error), 503) from error
        raise UploadError("invalid_image", str(error)) from error

    proc_time = (time.perf_counter() - t0) * 1000.0

    class_name = str(result["class_name"])
    analysis = get_fruit_analysis(class_name)

    prediction: dict[str, Any] = {
        "filename": clean_name,
        "predicted_class": int(result["class_id"]),
        "class_name": class_name,
        "class_name_vi": analysis.get("label_vi") or localized_class_name(class_name),
        "fruit_type": str(result.get("fruit", analysis.get("fruit_type", ""))),
        "fruit_type_vi": analysis.get("fruit_type_vi", ""),
        "ripeness": str(result.get("ripeness", analysis.get("ripeness", ""))),
        "ripeness_vi": analysis.get("ripeness_vi", ""),
        "quality": analysis.get("quality", "Đạt chuẩn"),
        "quality_code": analysis.get("quality_code", "good"),
        "recommendation": analysis.get("recommendation", ""),
        "processing_time_ms": float(result.get("processing_time_ms", proc_time)),
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
