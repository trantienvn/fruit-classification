"""Service dự đoán chất lượng trái cây theo bảng số liệu (Machine Learning)."""

from __future__ import annotations

import logging
from typing import Any

from src_tabular.predict_tabular import FruitQualityClassifierTabular

logger = logging.getLogger(__name__)


class TabularPredictionError(ValueError):
    """Lỗi khi xử lý dữ liệu bảng."""

    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def predict_tabular_metrics(
    classifier: FruitQualityClassifierTabular | None,
    features: dict[str, float],
) -> dict[str, Any]:
    """Kiểm tra và thực hiện dự đoán chất lượng từ 7 thuộc tính."""
    if classifier is None or not classifier.is_ready:
        raise TabularPredictionError(
            code="model_not_ready",
            message="Mô hình học máy dạng bảng chưa sẵn sàng. Hãy kiểm tra models_tabular/.",
            status_code=503,
        )

    try:
        result = classifier.predict(features)
        return result
    except Exception as error:
        logger.exception("Lỗi khi dự đoán bảng dữ liệu: %s", error)
        raise TabularPredictionError(
            code="prediction_failed",
            message=f"Lỗi khi tính toán dự đoán: {error}",
            status_code=500,
        ) from error
