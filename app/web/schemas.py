"""Mô hình response của REST API (hỗ trợ cả Deep Learning ảnh và Học máy bảng số liệu)."""

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
    dl_model_ready: bool = False
    tabular_model_ready: bool = False
    model_name: str | None = None
    dl_model_name: str | None = None
    tabular_model_name: str | None = None


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
        description="Tên lớp đầy đủ bằng tiếng Việt, ví dụ: Cam · Hỏng."
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


# ---------------------------------------------------------------------------
# Schemas cho Bản Học Máy theo Bảng số liệu (Tabular Fruit Quality)
# ---------------------------------------------------------------------------

class TabularPredictionRequest(BaseModel):
    """Yêu cầu đánh giá chất lượng trái cây theo 7 thuộc tính."""

    size: float = Field(default=0.0, description="Kích thước quả (Size)")
    weight: float = Field(default=0.0, description="Khối lượng quả (Weight)")
    sweetness: float = Field(default=0.0, description="Độ ngọt (Sweetness)")
    crunchiness: float = Field(default=0.0, description="Độ giòn (Crunchiness)")
    juiciness: float = Field(default=0.0, description="Độ mọng nước (Juiciness)")
    acidity: float = Field(default=0.0, description="Độ axit (Acidity)")
    ripeness: float = Field(default=0.0, description="Độ chín (Ripeness)")


class TabularFeatureAssessment(BaseModel):
    feature: str
    feature_vi: str
    value: float
    status: str


class TabularPredictionData(BaseModel):
    predicted_class: str = Field(description="'Good' hoặc 'Bad'")
    label_vi: str = Field(description="'Tốt (Đạt chuẩn)' hoặc 'Kém (Không đạt)'")
    confidence: float
    probabilities: dict[str, float]
    inputs: dict[str, float]
    inputs_vi: dict[str, float]
    assessments: list[TabularFeatureAssessment]
    model_name: str


class TabularPredictionResponse(BaseModel):
    success: bool = True
    data: TabularPredictionData
