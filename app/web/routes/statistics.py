"""Thống kê từ các lượt dự đoán đã lưu."""

from __future__ import annotations

from fastapi import APIRouter, Request

from src.dataset import CLASS_NAMES

from app.web.schemas import StatisticsResponse

router = APIRouter(tags=["statistics"])


@router.get(
    "/statistics", response_model=StatisticsResponse, summary="Thống kê lịch sử thật"
)
def get_statistics(request: Request) -> StatisticsResponse:
    class_names = getattr(request.app.state, "class_names", list(CLASS_NAMES))
    result = request.app.state.history_service.statistics(class_names)
    return StatisticsResponse(data=result)
