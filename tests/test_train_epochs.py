from __future__ import annotations

import csv

import joblib
import numpy as np
import pytest

from src.train_epochs import train_epochs


def test_train_epochs_saves_best_model_history_and_plot(tmp_path):
    rng = np.random.default_rng(42)
    centers = np.array([[-2.0, -2.0], [2.0, 2.0]])
    X_train = np.vstack(
        [rng.normal(center, 0.4, size=(12, 2)) for center in centers]
    )
    y_train = np.repeat([0, 1], 12)
    X_val = np.vstack(
        [rng.normal(center, 0.4, size=(5, 2)) for center in centers]
    )
    y_val = np.repeat([0, 1], 5)
    report_dir = tmp_path / "reports" / "epoch-runs" / "run-1"
    model_path = tmp_path / "models" / "epoch-runs" / "run-1" / "classifier.joblib"

    summary = train_epochs(
        X_train,
        y_train,
        X_val,
        y_val,
        report_dir=report_dir,
        model_path=model_path,
        epochs=4,
        learning_rate=0.01,
        use_pca=False,
        class_names=("Apple Unripe", "Apple Ripe"),
    )

    assert summary["epochs_requested"] == 4
    assert 1 <= summary["best_epoch"] <= 4
    assert summary["best_val_macro_f1"] == pytest.approx(1.0)
    assert summary["class_names"] == ["Apple Unripe", "Apple Ripe"]
    assert model_path.is_file()
    assert (report_dir / "epoch_summary.json").is_file()
    plot_path = report_dir / "epoch_training.png"
    assert plot_path.is_file()
    assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    with (report_dir / "epoch_history.csv").open(
        newline="", encoding="utf-8-sig"
    ) as file:
        history = list(csv.DictReader(file))
    assert len(history) == 4
    assert {"train_loss", "val_loss", "train_macro_f1", "val_macro_f1"} <= set(history[0])

    model = joblib.load(model_path)
    assert np.array_equal(model.predict(X_val), y_val)


def test_train_epochs_rejects_validation_class_missing_from_train(tmp_path):
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([0, 0, 1, 1])
    X_val = np.array([[4.0], [5.0]])
    y_val = np.array([1, 2])

    with pytest.raises(ValueError, match="lớp không có"):
        train_epochs(
            X_train,
            y_train,
            X_val,
            y_val,
            report_dir=tmp_path / "reports",
            model_path=tmp_path / "model.joblib",
            epochs=2,
            use_pca=False,
        )


def test_epoch_cli_creates_a_new_run_directory_each_time(tmp_path, monkeypatch):
    from src.train_epochs import main

    rng = np.random.default_rng(7)
    X_train = np.vstack([rng.normal(-1, 0.2, (6, 2)), rng.normal(1, 0.2, (6, 2))])
    y_train = np.repeat([0, 1], 6)
    X_val = np.vstack([rng.normal(-1, 0.2, (3, 2)), rng.normal(1, 0.2, (3, 2))])
    y_val = np.repeat([0, 1], 3)
    monkeypatch.setattr(
        "src.train_epochs._load_data",
        lambda *args, **kwargs: (
            X_train, y_train, X_val, y_val, X_val, y_val, [], [], {},
            ("Apple Unripe", "Apple Ripe"), (2, 1), "folder",
        ),
    )
    data_dir = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (data_dir / split).mkdir(parents=True)
    model_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    args = [
        "--data-dir", str(data_dir),
        "--model-dir", str(model_dir),
        "--reports-dir", str(reports_dir),
        "--epochs", "2",
        "--no-pca",
    ]

    assert main(args) == 0
    assert main(args) == 0

    report_runs = list((reports_dir / "epoch-runs").iterdir())
    model_runs = list((model_dir / "epoch-runs").iterdir())
    assert len(report_runs) == len(model_runs) == 2
    assert all((run / "epoch_training.png").is_file() for run in report_runs)
    assert all((run / "classifier.joblib").is_file() for run in model_runs)
