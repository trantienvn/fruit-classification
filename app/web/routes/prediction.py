"""Nhận file ảnh và trả kết quả từ FruitClassifier."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.web.schemas import ErrorResponse, PredictionResponse
from app.web.services.prediction_service import UploadError, predict_upload

router = APIRouter(tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
    summary="Phân loại ảnh trái cây",
    responses={
        413: {"model": ErrorResponse, "description": "Ảnh vượt quá giới hạn dung lượng"},
        415: {"model": ErrorResponse, "description": "Định dạng ảnh không được hỗ trợ"},
        422: {"model": ErrorResponse, "description": "Tệp ảnh không hợp lệ"},
        503: {"model": ErrorResponse, "description": "Model chưa sẵn sàng"},
    },
)
def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None or not classifier.is_ready:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_not_ready", "message": "Model chưa sẵn sàng."},
        )

    max_bytes = request.app.state.settings.max_upload_bytes
    image_bytes = file.file.read(max_bytes + 1)
    try:
        prediction = predict_upload(
            classifier,
            filename=file.filename,
            image_bytes=image_bytes,
            max_upload_bytes=max_bytes,
        )
    except UploadError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail={"code": error.code, "message": str(error)},
        ) from error

    saved = request.app.state.history_service.add(prediction)
    if "probabilities" in prediction:
        saved["probabilities"] = prediction["probabilities"]
    return PredictionResponse(data=saved)
