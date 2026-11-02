"""Kiểm thử REST API không cần dataset hoặc model đã huấn luyện."""

from __future__ import annotations

import json
import sqlite3
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.web.config import Settings
from app.web.main import create_app
from src.dataset import CLASS_NAMES

class StubClassifier:
    """Thay classifier ở biên HTTP để kiểm thử độc lập artifact model."""

    is_ready = False
    supports_probability_output = True

    def __init__(self, _model_dir: Path) -> None:
        self.is_ready = False

    def load(self) -> "StubClassifier":
        self.is_ready = True
        return self

    def get_info(self) -> dict:
        return {
            "model_name": "SVM test stub",
            "n_classes": len(CLASS_NAMES),
            "class_names": list(CLASS_NAMES),
            "n_features": 51,
            "use_pca": False,
            "val_metrics": {},
            "test_metrics": {"accuracy": 0.9, "macro_f1": 0.88},
            "n_train": 10,
            "status": "Sẵn sàng",
        }

    def predict_from_bytes(self, _image_bytes: bytes) -> dict:
        probabilities = {name: 0.0 for name in CLASS_NAMES}
        probabilities["Banana Ripe"] = 0.75
        probabilities["Banana Spoiled"] = 0.25
        return {
            "class_id": 4,
            "class_name": "Banana Ripe",
            "fruit": "Banana",
            "ripeness": "Ripe",
            "confidence": 0.75,
            "probabilities": probabilities,
            "processing_time_ms": 4.2,
        }


def make_client(tmp_path: Path, classifier_factory=StubClassifier) -> TestClient:
    app_settings = Settings(
        model_dir=tmp_path / "models",
        history_db=tmp_path / "history.sqlite3",
        frontend_root=Path(__file__).parents[1] / "app" / "web" / "frontend",
        reports_root=tmp_path / "reports",
        max_upload_bytes=10 * 1024 * 1024,
    )
    return TestClient(create_app(app_settings, classifier_factory))


def make_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 32), color=(190, 120, 55)).save(output, format="PNG")
    return output.getvalue()


def test_health_and_frontend_pages_are_served(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["data"]["status"] == "ready"
        assert health.json()["data"]["model_name"] == "SVM test stub"

        assert "Vườn chín" in client.get("/").text
        assert "Chọn ảnh trái cây" in client.get("/pages/recognition.html").text
        assert client.get("/static/css/main.css").status_code == 200
        assert client.get("/pages/not-allowed.html").status_code == 404


def test_feature_importance_report_is_served(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    reports_root.mkdir()
    expected_image = make_png()
    (reports_root / "feature_importance.png").write_bytes(expected_image)

    with make_client(tmp_path) as client:
        response = client.get("/reports/feature_importance.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == expected_image


def test_openapi_describes_api_contract(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        schema = client.get("/openapi.json").json()
        snapshot_path = Path(__file__).parents[1] / "app" / "web" / "openapi.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        assert schema == snapshot
        assert {"/api/health", "/api/predict", "/api/model-info", "/api/history", "/api/statistics"} <= set(schema["paths"])
        upload = schema["paths"]["/api/predict"]["post"]["requestBody"]
        assert "multipart/form-data" in upload["content"]


def test_predict_persists_metadata_only_and_updates_statistics(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.post(
            "/api/predict",
            files={"file": ("folder\\banana.png", make_png(), "image/png")},
        )
        assert response.status_code == 200
        prediction = response.json()["data"]
        assert prediction["predicted_class"] == 4
        assert prediction["fruit_type"] == "Banana"
        assert prediction["confidence"] == 0.75
        assert prediction["class_name_vi"] == "Chuối · Chín"
        assert "probabilities" in prediction
        assert prediction["filename"] == "banana.png"

        history = client.get("/api/history").json()["data"]
        assert history["total"] == 1
        assert history["items"][0]["class_name"] == "Banana Ripe"

        statistics = client.get("/api/statistics").json()["data"]
        assert statistics["total_predictions"] == 1
        assert statistics["unique_classes"] == 1
        assert next(row for row in statistics["by_class"] if row["class_name"] == "Banana Ripe")["count"] == 1
        assert sum(row["count"] for row in statistics["by_day"]) == 1

        with sqlite3.connect(tmp_path / "history.sqlite3") as connection:
            columns = connection.execute("PRAGMA table_info(predictions)").fetchall()
        assert "image_bytes" not in {row[1] for row in columns}


def test_upload_rejects_unsupported_invalid_and_oversized_files(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        unsupported = client.post(
            "/api/predict", files={"file": ("fruit.gif", make_png(), "image/gif")}
        )
        assert unsupported.status_code == 415
        assert unsupported.json()["error"]["code"] == "unsupported_format"

        invalid = client.post(
            "/api/predict", files={"file": ("broken.png", b"not an image", "image/png")}
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "invalid_image"

        oversized_settings = Settings(
            model_dir=tmp_path / "models",
            history_db=tmp_path / "small-limit.sqlite3",
            frontend_root=Path(__file__).parents[1] / "app" / "web" / "frontend",
            reports_root=tmp_path / "reports",
            max_upload_bytes=24,
        )
        oversized_client = TestClient(create_app(oversized_settings, StubClassifier))
        with oversized_client:
            oversized = oversized_client.post(
                "/api/predict", files={"file": ("large.png", make_png(), "image/png")}
            )
        assert oversized.status_code == 413
        assert oversized.json()["error"]["code"] == "file_too_large"


def test_history_filters_and_can_be_cleared(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        for filename in ("banana.png", "mango.png"):
            response = client.post(
                "/api/predict", files={"file": (filename, make_png(), "image/png")}
            )
            assert response.status_code == 200

        filtered = client.get("/api/history", params={"q": "mango", "class_name": "Banana Ripe"}).json()["data"]
        assert filtered["total"] == 1
        assert filtered["items"][0]["filename"] == "mango.png"

        deleted = client.delete("/api/history").json()["data"]["deleted"]
        assert deleted == 2
        assert client.get("/api/history").json()["data"]["total"] == 0


def test_statistics_keeps_legacy_history_labels(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        client.app.state.history_service.add(
            {
                "filename": "old-apple.png",
                "predicted_class": 0,
                "class_name": "Apple Ripe",
                "fruit_type": "Apple",
                "ripeness": "Ripe",
                "confidence": 0.8,
                "processing_time_ms": 3.0,
            }
        )

        statistics = client.get("/api/statistics").json()["data"]

    by_fruit = {row["fruit"]: row["count"] for row in statistics["by_fruit"]}
    by_class = {row["class_name"]: row["count"] for row in statistics["by_class"]}
    assert by_fruit["Apple"] == 1
    assert by_class["Apple Ripe"] == 1


def test_missing_model_leaves_api_alive_and_reports_unavailable(tmp_path: Path) -> None:
    def missing_model(_model_dir: Path):
        raise FileNotFoundError("model missing")

    with make_client(tmp_path, missing_model) as client:
        assert client.get("/api/health").json()["data"]["status"] == "degraded"
        prediction = client.post(
            "/api/predict", files={"file": ("fruit.png", make_png(), "image/png")}
        )
        assert prediction.status_code == 503
        assert prediction.json()["error"]["code"] == "model_not_ready"
        assert client.get("/api/model-info").status_code == 503


def test_unavailable_probability_api_omits_confidence_and_probabilities(tmp_path: Path) -> None:
    class UncalibratedClassifier(StubClassifier):
        supports_probability_output = False

    with make_client(tmp_path, UncalibratedClassifier) as client:
        response = client.post(
            "/api/predict", files={"file": ("fruit.png", make_png(), "image/png")}
        )
        assert response.status_code == 200
        result = response.json()["data"]
        assert "confidence" not in result
        assert "probabilities" not in result
