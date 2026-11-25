"""Kiểm thử ứng dụng FastAPI Web phục vụ mô hình Học Sâu (Deep Learning)."""

import io
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.web.main import app


def create_sample_fruit_image() -> io.BytesIO:
    img = Image.new("RGB", (224, 224), color=(255, 140, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_web_app():
    with TestClient(app) as client:
        # 1. Health check
        res_health = client.get("/api/health")
        assert res_health.status_code == 200, res_health.text
        data_health = res_health.json()["data"]
        print("HEALTH STATUS:", data_health)
        assert data_health["status"] in ("ready", "degraded")
        assert data_health["n_classes"] == 9

        # 2. Model info
        res_info = client.get("/api/model-info")
        assert res_info.status_code == 200, res_info.text
        data_info = res_info.json()["data"]
        print("MODEL INFO:", data_info.get("model_name"), "Backbone:", data_info.get("backbone"))
        assert "dual_attributes" in data_info
        assert len(data_info["dual_attributes"]["fruit_types"]) == 3

        # 3. Predict image (if classifier ready)
        img_buf = create_sample_fruit_image()
        res_predict = client.post(
            "/api/predict",
            files={"file": ("test_orange.jpg", img_buf, "image/jpeg")},
        )
        if res_predict.status_code == 200:
            pred_data = res_predict.json()["data"]
            print("PREDICTION RESULT:", pred_data)
            assert "class_name" in pred_data
            assert "quality" in pred_data
            assert "quality_code" in pred_data
            assert "recommendation" in pred_data
            assert "probabilities" in pred_data
        else:
            print("Predict response (model might need weights):", res_predict.status_code, res_predict.text)
            assert res_predict.status_code in (200, 503)


if __name__ == "__main__":
    test_web_app()
