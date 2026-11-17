"""Trạng thái tiến trình, mô hình Deep Learning và kho lưu trữ lịch sử."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.web.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Kiểm tra trạng thái API")
def health(request: Request) -> HealthResponse:
    classifier = getattr(request.app.state, "classifier", None)
    model_ready = classifier is not None and getattr(classifier, "is_ready", False)

    model_name = None
    backbone = None
    device = None
    n_classes = 9

    if model_ready:
        if hasattr(classifier, "get_info"):
            info = classifier.get_info()
            model_name = info.get("model_name", "CNN (MobileNetV3) + MLP")
            backbone = info.get("backbone", "mobilenetv3_small")
            n_classes = info.get("n_classes", 9)
        device = getattr(classifier, "device", "auto")

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
            "model_name": model_name or "MobileNetV3-Small + PyTorch MLP",
            "backbone": backbone or "mobilenetv3_small",
            "device": str(device) if device else "CUDA / CPU",
            "n_classes": n_classes,
        }
    )
