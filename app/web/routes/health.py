"""Trạng thái tiến trình, model và kho lịch sử."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.web.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Kiểm tra trạng thái API")
def health(request: Request) -> HealthResponse:
    classifier = getattr(request.app.state, "classifier", None)
    model_ready = classifier is not None and classifier.is_ready
    model_name = None
    if model_ready:
        model_name = classifier.get_info().get("model_name")

    history_ready = True
    try:
        request.app.state.history_service.ping()
    except Exception:
        history_ready = False

    status = "ready" if model_ready and history_ready else "degraded"
    return HealthResponse(
        data={
            "status": status,
            "model_ready": model_ready,
            "history_ready": history_ready,
            "model_name": model_name,
        }
    )
