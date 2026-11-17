"""Kiểm tra Vercel Serverless Entry point toàn diện cho ứng dụng Học Sâu (Deep Learning)."""

import io
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.index import app
from fastapi.testclient import TestClient


def create_dummy_image_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(255, 128, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_vercel_app():
    with TestClient(app) as client:
        # 1. Kiểm tra trang chủ /
        res_root = client.get("/")
        print("GET / status:", res_root.status_code)
        assert res_root.status_code == 200
        assert "<html" in res_root.text

        # 2. Kiểm tra khi Vercel rewrite đến /api/index
        res_api_index = client.get("/api/index")
        print("GET /api/index status:", res_api_index.status_code)
        assert res_api_index.status_code == 200
        assert "<html" in res_api_index.text

        # 3. Kiểm tra Vercel x-matched-path header
        res_matched = client.get("/api/index", headers={"x-matched-path": "/"})
        print("GET /api/index with x-matched-path='/' status:", res_matched.status_code)
        assert res_matched.status_code == 200
        assert "<html" in res_matched.text

        # 4. Kiểm tra trang phụ /pages/recognition.html
        res_recog = client.get("/pages/recognition.html")
        print("GET /pages/recognition.html status:", res_recog.status_code)
        assert res_recog.status_code == 200

        # 5. Kiểm tra file tĩnh /static/css/main.css
        res_css = client.get("/static/css/main.css")
        print("GET /static/css/main.css status:", res_css.status_code)
        assert res_css.status_code == 200

        # 6. Kiểm tra API health
        res_health = client.get("/api/health")
        print("GET /api/health status:", res_health.status_code)
        assert res_health.status_code == 200
        health_data = res_health.json()["data"]
        print("Health data:", health_data)
        assert health_data["status"] in ("ready", "degraded")

        # 7. Kiểm tra API model-info
        res_info = client.get("/api/model-info")
        print("GET /api/model-info status:", res_info.status_code)
        assert res_info.status_code == 200
        info_data = res_info.json()["data"]
        print("Model info:", info_data.get("model_name"), "Backbone:", info_data.get("backbone"))
        assert "dual_attributes" in info_data

        print("\n=> TẤT CẢ ROUTE VÀ TRANG TĨNH ĐỀU TRẢ VỀ HTTP 200! SẴN SÀNG HOÀN TOÀN TRÊN VERCEL!")


if __name__ == "__main__":
    test_vercel_app()
