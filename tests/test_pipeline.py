import pytest

from src.evaluate import evaluate_predictions
from src.predict import checkpoint_available
from src.train import require_dataset


def test_require_dataset_reports_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        require_dataset(tmp_path / "missing")


def test_checkpoint_available(tmp_path):
    checkpoint = tmp_path / "model.pth"
    assert not checkpoint_available(checkpoint)
    checkpoint.write_bytes(b"placeholder")
    assert checkpoint_available(checkpoint)


def test_evaluate_predictions_returns_basic_metrics():
    metrics = evaluate_predictions(["ripe", "unripe"], ["ripe", "ripe"])
    assert metrics["accuracy"] == 0.5
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert "ripe" in metrics["report"]

