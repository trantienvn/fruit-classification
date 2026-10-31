"""Metadata lấy từ model/config hiện đang nạp."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request

from app.web.schemas import ErrorResponse, ModelInfoResponse

router = APIRouter(tags=["model"])


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Thông tin model hiện tại",
    responses={503: {"model": ErrorResponse, "description": "Model chưa sẵn sàng"}},
)
def model_info(request: Request) -> ModelInfoResponse:
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None or not classifier.is_ready:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_not_ready", "message": "Model chưa sẵn sàng."},
        )
    info = classifier.get_info()
    config_path = request.app.state.settings.model_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        config = {}
    for key in ("crop_size", "n_val", "n_test", "gridsearch"):
        if key in config:
            info[key] = config[key]
    return ModelInfoResponse(data=info)
