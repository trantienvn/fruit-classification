"""Metadata mô hình Deep Learning (MobileNetV3 + PyTorch MLP), kiến trúc và các chỉ số thực nghiệm."""

from __future__ import annotations

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request

from app.web.labels import FULL_CLASS_MAP_VI
from app.web.schemas import ErrorResponse, ModelInfoResponse

router = APIRouter(tags=["model"])


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Thông tin mô hình học sâu Deep Learning",
    responses={503: {"model": ErrorResponse, "description": "Model chưa sẵn sàng"}},
)
def model_info(request: Request) -> ModelInfoResponse:
    classifier = getattr(request.app.state, "classifier", None)
    settings = request.app.state.settings

    info: dict = {
        "model_type": "deep_learning",
        "model_name": "CNN (MobileNetV3-Small) + PyTorch MLP",
        "backbone": "mobilenetv3_small",
        "embedding_dim": 576,
        "input_size": [224, 224],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "dual_attributes": {
            "fruit_types": ["Cam", "Chuối", "Xoài"],
            "ripeness_stages": ["Chín", "Chưa chín (Xanh)", "Hỏng / Quá chín"],
            "quality_grades": ["Đạt chuẩn (Loại 1)", "Đạt chuẩn lưu kho (Loại 2)", "Không đạt chuẩn (Loại thải)"],
        },
    }

    if classifier is not None and getattr(classifier, "is_ready", False):
        if hasattr(classifier, "get_info"):
            info.update(classifier.get_info())

    # Đọc thông số thực nghiệm từ models/config.json
    config_path = settings.dl_model_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for key in (
                "crop_size",
                "test_accuracy",
                "test_f1",
                "total_epochs",
                "best_model",
                "device",
                "class_names",
                "n_classes",
                "use_pca",
                "pca_ratio",
            ):
                if key in config:
                    info[key] = config[key]
        except (OSError, json.JSONDecodeError):
            pass

    # Gán tên hiển thị tiếng Việt cho các lớp
    class_names = info.get("class_names", [])
    info["classes_vi"] = [
        {"id": idx, "name": c, "label_vi": FULL_CLASS_MAP_VI.get(c, c)}
        for idx, c in enumerate(class_names)
    ]

    # Kiểm tra các biểu đồ báo cáo có sẵn
    reports_dir = settings.reports_root
    reports = []
    if (reports_dir / "loss_curve.png").exists():
        reports.append({
            "title": "Sơ đồ xu hướng Loss & Accuracy qua từng Epoch",
            "url": "/reports/loss_curve.png",
            "type": "loss_curve",
            "description": "Đường cong Cross-Entropy Loss và Validation Accuracy theo từng chu kỳ huấn luyện.",
        })
    if (reports_dir / "epoch_training.png").exists():
        reports.append({
            "title": "Biểu đồ đánh giá huấn luyện & Tốc độ hội tụ",
            "url": "/reports/epoch_training.png",
            "type": "training_metrics",
            "description": "So sánh chi tiết tốc độ hội tụ và phân bố độ chính xác trên tập Train và Validation.",
        })
    info["reports"] = reports

    return ModelInfoResponse(data=info)
