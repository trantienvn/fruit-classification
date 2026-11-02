"""Mô hình response của REST API."""

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


class HealthResponse(BaseModel):
    success: bool = True
    data: HealthData


class PredictionData(BaseModel):
    id: int
    filename: str
    fruit_type: str
    ripeness: str
    class_id: int = Field(alias="predicted_class")
    class_name: str
    class_name_vi: str = Field(
        description="Tên lớp đầy đủ bằng tiếng Việt, ví dụ: Chuối · Quá chín."
    )
    confidence: float | None = Field(
        default=None,
        description="Chỉ có khi model cung cấp xác suất dự đoán thật.",
    )
    processing_time_ms: float
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="Chỉ có khi classifier hỗ trợ predict_proba.",
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
