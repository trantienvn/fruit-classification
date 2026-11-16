"""Kiểm tra Vercel Serverless Entry point."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.index import app
from fastapi.testclient import TestClient

def test_vercel_app():
    with TestClient(app) as client:
        res = client.get("/api/health")
        print("Health status code:", res.status_code)
        print("Health body:", res.json())
        assert res.status_code == 200

        res_root = client.get("/")
        print("Root UI status code:", res_root.status_code)
        assert res_root.status_code == 200

        # Kiểm tra API Tabular ML
        res_tab = client.post(
            "/api/tabular/predict",
            json={
                "size": 1.0,
                "weight": 1.0,
                "sweetness": 1.5,
                "crunchiness": 1.0,
                "juiciness": 1.0,
                "acidity": -0.5,
                "ripeness": 1.0,
            },
        )
        print("Tabular predict status code:", res_tab.status_code)
        print("Tabular predict body:", res_tab.json()["data"]["predicted_class"])
        assert res_tab.status_code == 200

        print("\n=> ENTRYPOINT CHO VERCEL ĐÃ HOẠT ĐỘNG HOÀN TOÀN CHÍNH XÁC VÀ SẴN SÀNG TRIỂN KHAI!")

if __name__ == "__main__":
    test_vercel_app()
