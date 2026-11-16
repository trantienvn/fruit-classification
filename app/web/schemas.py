"""Mô hình schema Pydantic cho REST API ứng dụng Học Sâu (Deep Learning) Phân Loại Trái Cây."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ApiError(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ApiError


class HealthData(BaseModel):
    status: str
    model_ready: bool
    history_ready: bool
    model_name: str | None = None
    backbone: str | None = None
    device: str | None = None
    n_classes: int = 9


class HealthResponse(BaseModel):
    success: bool = True
    data: HealthData


class PredictionData(BaseModel):
    id: int
    filename: str
    fruit_type: str
    fruit_type_vi: str = Field(default="", description="Tên tiếng Việt loại trái cây (Cam, Chuối, Xoài)")
    ripeness: str
    ripeness_vi: str = Field(default="", description="Độ chín tiếng Việt (Chín mọng, Chưa chín, Hỏng)")
    quality: str = Field(default="", description="Đánh giá chất lượng thương phẩm (Đạt chuẩn / Kém)")
    quality_code: str = Field(default="good", description="'good', 'unripe', hoặc 'bad'")
    recommendation: str = Field(default="", description="Lời khuyên bảo quản và tiêu dùng")
    class_id: int = Field(alias="predicted_class")
    class_name: str
    class_name_vi: str = Field(
        description="Tên lớp đầy đủ bằng tiếng Việt, ví dụ: Cam · Chín mọng."
    )
    confidence: float | None = Field(
        default=None,
        description="Độ tin cậy xác suất dự đoán cao nhất (0.0 - 1.0).",
    )
    processing_time_ms: float
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="Phân bố xác suất trên toàn bộ 9 lớp.",
    )
    created_at: str


class PredictionResponse(BaseModel):
    success: bool = True
    data: PredictionData


class HistoryData(BaseModel):
    items: list[PredictionData]
    total: int
    limit: int
    offset: int


class HistoryResponse(BaseModel):
    success: bool = True
    data: HistoryData


class StatisticsData(BaseModel):
    total_predictions: int
    unique_classes: int
    latest_at: str | None = None
    by_class: list[dict[str, Any]]
    by_fruit: list[dict[str, Any]]
    by_ripeness: list[dict[str, Any]]
    by_day: list[dict[str, Any]]


class StatisticsResponse(BaseModel):
    success: bool = True
    data: StatisticsData


class ModelInfoResponse(BaseModel):
    success: bool = True
    data: dict[str, Any]


class DeleteHistoryData(BaseModel):
    deleted: int


class DeleteHistoryResponse(BaseModel):
    success: bool = True
    data: DeleteHistoryData
