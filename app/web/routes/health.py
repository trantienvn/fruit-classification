"""Trạng thái tiến trình, các mô hình (Deep Learning & Tabular ML) và kho lịch sử."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.web.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Kiểm tra trạng thái API")
def health(request: Request) -> HealthResponse:
    classifier = getattr(request.app.state, "classifier", None)
    tabular_classifier = getattr(request.app.state, "tabular_classifier", None)

    dl_ready = classifier is not None and getattr(classifier, "is_ready", False)
    tabular_ready = tabular_classifier is not None and getattr(tabular_classifier, "is_ready", False)

    dl_model_name = None
    if dl_ready and hasattr(classifier, "get_info"):
        dl_model_name = classifier.get_info().get("model_name")

    tabular_model_name = None
    if tabular_ready and hasattr(tabular_classifier, "config"):
        tabular_model_name = tabular_classifier.config.get("model_name") or tabular_classifier.config.get("best_model")

    history_ready = True
    try:
        request.app.state.history_service.ping()
    except Exception:
        history_ready = False

    status = "ready" if (dl_ready or tabular_ready) and history_ready else "degraded"

    return HealthResponse(
        data={
            "status": status,
            "model_ready": dl_ready,
            "history_ready": history_ready,
            "dl_model_ready": dl_ready,
            "tabular_model_ready": tabular_ready,
            "model_name": dl_model_name or "MobileNetV3-Small + MLP",
            "dl_model_name": dl_model_name,
            "tabular_model_name": tabular_model_name,
        }
    )
