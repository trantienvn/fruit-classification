"""Route API đánh giá chất lượng trái cây theo số liệu dạng bảng (Machine Learning)."""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, Request

from app.web.schemas import (
    ErrorResponse,
    TabularPredictionData,
    TabularPredictionRequest,
    TabularPredictionResponse,
)
from app.web.services.tabular_prediction_service import (
    TabularPredictionError,
    predict_tabular_metrics,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tabular", tags=["tabular"])


@router.post(
    "/predict",
    response_model=TabularPredictionResponse,
    summary="Đánh giá chất lượng trái cây theo bảng số liệu thuộc tính",
    responses={
        422: {"model": ErrorResponse, "description": "Dữ liệu đầu vào không hợp lệ"},
        503: {"model": ErrorResponse, "description": "Mô hình học máy dạng bảng chưa sẵn sàng"},
    },
)
def predict_tabular(
    request: Request,
    payload: TabularPredictionRequest,
) -> TabularPredictionResponse:
    classifier = getattr(request.app.state, "tabular_classifier", None)
    if classifier is None or not classifier.is_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_not_ready",
                "message": "Mô hình học máy dạng bảng chưa sẵn sàng.",
            },
        )

    try:
        features = {
            "Size": payload.size,
            "Weight": payload.weight,
            "Sweetness": payload.sweetness,
            "Crunchiness": payload.crunchiness,
            "Juiciness": payload.juiciness,
            "Acidity": payload.acidity,
            "Ripeness": payload.ripeness,
        }
        result = predict_tabular_metrics(classifier, features)
        return TabularPredictionResponse(
            success=True,
            data=TabularPredictionData(**result),
        )
    except TabularPredictionError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail={"code": error.code, "message": error.message},
        ) from error
    except Exception as error:
        logger.exception("Lỗi xử lý dự đoán bảng: %s", error)
        raise HTTPException(
            status_code=500,
            detail={"code": "internal_error", "message": str(error)},
        ) from error


@router.get(
    "/info",
    summary="Lấy thông tin cấu hình mô hình dạng bảng",
)
def get_tabular_info(request: Request) -> dict:
    classifier = getattr(request.app.state, "tabular_classifier", None)
    if classifier is None or not classifier.is_ready:
        return {
            "ready": False,
            "message": "Mô hình dạng bảng chưa được tải",
        }
    return {
        "ready": True,
        "config": classifier.config,
        "feature_names": classifier.feature_names,
        "classes": classifier.classes,
    }
