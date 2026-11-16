"""Metadata lấy từ các model/config hiện đang nạp (Deep Learning ảnh & Machine Learning bảng số liệu)."""

from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException, Request

from app.web.schemas import ErrorResponse, ModelInfoResponse

router = APIRouter(tags=["model"])


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Thông tin các mô hình hiện tại",
    responses={503: {"model": ErrorResponse, "description": "Model chưa sẵn sàng"}},
)
def model_info(request: Request) -> ModelInfoResponse:
    classifier = getattr(request.app.state, "classifier", None)
    tabular_classifier = getattr(request.app.state, "tabular_classifier", None)

    if (classifier is None or not getattr(classifier, "is_ready", False)) and (
        tabular_classifier is None or not getattr(tabular_classifier, "is_ready", False)
    ):
        raise HTTPException(
            status_code=503,
            detail={"code": "model_not_ready", "message": "Cả hai mô hình đều chưa sẵn sàng."},
        )

    info: dict = {}
    if classifier is not None and getattr(classifier, "is_ready", False):
        if hasattr(classifier, "get_info"):
            info = classifier.get_info()
        else:
            info = {"model_name": "FruitClassifier"}

        config_path = request.app.state.settings.model_dir / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for key in ("crop_size", "n_val", "n_test", "gridsearch", "test_accuracy", "test_f1", "device"):
                if key in config:
                    info[key] = config[key]
        except (OSError, json.JSONDecodeError):
            pass

    # Đính kèm thông tin mô hình học máy dạng bảng (Tabular ML)
    if tabular_classifier is not None and getattr(tabular_classifier, "is_ready", False):
        info["tabular_model"] = {
            "ready": True,
            "model_name": tabular_classifier.config.get("best_model", "SVM_RBF"),
            "features": tabular_classifier.feature_names,
            "feature_labels_vi": tabular_classifier.config.get("feature_labels_vi", {}),
            "classes": tabular_classifier.classes,
            "class_labels_vi": tabular_classifier.config.get("class_labels_vi", {}),
            "test_accuracy": tabular_classifier.config.get("test_accuracy"),
            "test_f1": tabular_classifier.config.get("test_f1"),
        }
    else:
        info["tabular_model"] = {"ready": False}

    return ModelInfoResponse(data=info)
