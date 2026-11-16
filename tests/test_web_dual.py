"""Kiểm thử hai phiên bản mô hình trên FastAPI Web App (DL Image & Tabular ML)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.web.main import app

def test_dual_mode_web():
    with TestClient(app) as client:
        # 1. Health check
        res_health = client.get("/api/health")
        assert res_health.status_code == 200, res_health.text
        data_health = res_health.json()["data"]
        print("HEALTH STATUS:", data_health)
        assert data_health["status"] in ("ready", "degraded")
        assert data_health["tabular_model_ready"] is True

        # 2. Model info
        res_info = client.get("/api/model-info")
        assert res_info.status_code == 200, res_info.text
        data_info = res_info.json()["data"]
        print("MODEL INFO:", data_info.get("model_name"), "Tabular:", data_info.get("tabular_model", {}).get("model_name"))
        assert "tabular_model" in data_info
        assert data_info["tabular_model"]["ready"] is True

        # 3. Tabular Prediction: Good quality
        payload_good = {
            "size": 1.2,
            "weight": 1.5,
            "sweetness": 2.1,
            "crunchiness": 1.4,
            "juiciness": 1.8,
            "acidity": -0.8,
            "ripeness": 1.1,
        }
        res_tab_good = client.post("/api/tabular/predict", json=payload_good)
        assert res_tab_good.status_code == 200, res_tab_good.text
        data_tab_good = res_tab_good.json()["data"]
        print("TABULAR PREDICT (Good):", data_tab_good["predicted_class"], data_tab_good["label_vi"], "Confidence:", data_tab_good["confidence"])
        assert data_tab_good["predicted_class"] == "Good"

        # 4. Tabular Prediction: Bad quality
        payload_bad = {
            "size": -1.5,
            "weight": -1.2,
            "sweetness": -2.0,
            "crunchiness": -1.5,
            "juiciness": -1.5,
            "acidity": 2.5,
            "ripeness": -1.5,
        }
        res_tab_bad = client.post("/api/tabular/predict", json=payload_bad)
        assert res_tab_bad.status_code == 200, res_tab_bad.text
        data_tab_bad = res_tab_bad.json()["data"]
        print("TABULAR PREDICT (Bad):", data_tab_bad["predicted_class"], data_tab_bad["label_vi"], "Confidence:", data_tab_bad["confidence"])
        assert data_tab_bad["predicted_class"] == "Bad"

        # 5. Tabular Info endpoint
        res_tab_info = client.get("/api/tabular/info")
        assert res_tab_info.status_code == 200
        print("TABULAR INFO:", res_tab_info.json()["ready"])

        # 6. Test Image Prediction qua Deep Learning API
        sample_img_path = Path("data/raw/images/test/-1_jpg.rf.d74fe43dc3420613d4d51e940c55d319.jpg")
        if sample_img_path.exists():
            with open(sample_img_path, "rb") as f:
                res_img = client.post(
                    "/api/predict",
                    files={"file": ("test_fruit.jpg", f, "image/jpeg")},
                )
            assert res_img.status_code == 200, res_img.text
            img_data = res_img.json()["data"]
            print("DL IMAGE PREDICT:", img_data["class_name"], "| Tiếng Việt:", img_data["class_name_vi"], "| Confidence:", img_data.get("confidence"))
            assert "class_name" in img_data

        print("\n=> TẤT CẢ TEST WEB CHO CẢ 2 BẢN MÔ HÌNH (ẢNH & BẢNG SỐ LIỆU) ĐÃ THÀNH CÔNG RỰC RỠ!")

if __name__ == "__main__":
    test_dual_mode_web()
