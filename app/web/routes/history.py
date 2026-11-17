"""Lịch sử metadata suy luận được lưu cục bộ."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from app.web.schemas import (
    DeleteHistoryResponse,
    HistoryResponse,
)

router = APIRouter(tags=["history"])


@router.get(
    "/history",
    response_model=HistoryResponse,
    response_model_exclude_none=True,
    summary="Đọc lịch sử dự đoán",
)
def get_history(
    request: Request,
    q: str | None = Query(default=None, max_length=120, description="Tìm theo tên tệp"),
    class_name: str | None = Query(default=None, max_length=80),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> HistoryResponse:
    items, total = request.app.state.history_service.list(
        query=q, class_name=class_name, limit=limit, offset=offset
    )
    return HistoryResponse(
        data={"items": items, "total": total, "limit": limit, "offset": offset}
    )


@router.delete(
    "/history", response_model=DeleteHistoryResponse, summary="Xóa lịch sử dự đoán"
)
def delete_history(request: Request) -> DeleteHistoryResponse:
    deleted = request.app.state.history_service.clear()
    return DeleteHistoryResponse(data={"deleted": deleted})
