"""Huấn luyện SGD theo epoch và lưu lịch sử từng lần chạy.

Chạy:
    python -m src.train_epochs --epochs 50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.train import _default_data_dir, _load_data


def _validate_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    learning_rate: float,
) -> np.ndarray:
    """Kiểm tra đầu vào và trả về danh sách lớp có trong tập train."""
    if epochs < 1:
        raise ValueError("epochs phải lớn hơn 0.")
    if learning_rate <= 0:
        raise ValueError("learning_rate phải lớn hơn 0.")
    if (
        X_train.ndim != 2
        or X_val.ndim != 2
        or X_train.shape[1] != X_val.shape[1]
        or len(X_train) != len(y_train)
        or len(X_val) != len(y_val)
        or not len(y_train)
        or not len(y_val)
    ):
        raise ValueError("Dữ liệu train/validation phải là ma trận hợp lệ, không rỗng.")

    classes = np.unique(y_train)
    if len(classes) < 2:
        raise ValueError("Tập train cần có ít nhất hai lớp.")
    if not np.isin(y_val, classes).all():
        raise ValueError("Tập validation chứa lớp không có trong tập train.")
    return classes


def _score_epoch(
    model: SGDClassifier,
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
) -> tuple[float, float, float]:
    probabilities = model.predict_proba(X)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    return (
        float(log_loss(y, probabilities, labels=classes)),
        float(accuracy_score(y, predictions)),
        float(f1_score(y, predictions, labels=classes, average="macro", zero_division=0)),
    )


def _save_epoch_history(history: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "epoch_history.csv"
    fieldnames = list(history[0])
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    figure, (f1_axis, loss_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    f1_axis.plot(epochs, [row["train_macro_f1"] for row in history], label="Train")
    f1_axis.plot(epochs, [row["val_macro_f1"] for row in history], label="Validation")
    f1_axis.set(title="Macro-F1 theo epoch", xlabel="Epoch", ylabel="Macro-F1")
    f1_axis.set_ylim(0, 1)
    f1_axis.grid(alpha=0.25)
    f1_axis.legend()

    loss_axis.plot(epochs, [row["train_loss"] for row in history], label="Train")
    loss_axis.plot(epochs, [row["val_loss"] for row in history], label="Validation")
    loss_axis.set(title="Log loss theo epoch", xlabel="Epoch", ylabel="Log loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()

    figure.tight_layout()
    figure.savefig(output_dir / "epoch_training.png", dpi=150)
    plt.close(figure)


def train_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    report_dir: Path,
    model_path: Path,
    epochs: int = 50,
    learning_rate: float = 0.01,
    use_pca: bool = True,
    random_state: int = 42,
    class_names: tuple[str, ...] | None = None,
) -> dict:
    """Huấn luyện từng epoch, chọn model tốt nhất theo validation Macro-F1."""
    X_train = np.asarray(X_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    classes = _validate_data(
        X_train, y_train, X_val, y_val, epochs, learning_rate
    )
    if class_names is not None and len(class_names) != len(classes):
        raise ValueError("Số class_names phải bằng số lớp trong tập train.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    pca: PCA | None = None
    if use_pca:
        pca = PCA(n_components=0.95, random_state=random_state)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_val_scaled = pca.transform(X_val_scaled)

    classifier = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=learning_rate,
        shuffle=False,
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state)
    history = []
    best_model = None
    best_epoch = 0
    best_score = (-1.0, float("-inf"))

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y_train))
        if epoch == 1:
            classifier.partial_fit(
                X_train_scaled[order],
                y_train[order],
                classes=classes,
            )
        else:
            classifier.partial_fit(X_train_scaled[order], y_train[order])

        train_loss, train_accuracy, train_macro_f1 = _score_epoch(
            classifier, X_train_scaled, y_train, classes
        )
        val_loss, val_accuracy, val_macro_f1 = _score_epoch(
            classifier, X_val_scaled, y_val, classes
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "train_macro_f1": train_macro_f1,
                "val_macro_f1": val_macro_f1,
            }
        )
        score = (val_macro_f1, -val_loss)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_model = deepcopy(classifier)

    if best_model is None:
        raise RuntimeError("Không tạo được checkpoint huấn luyện.")

    report_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    _save_epoch_history(history, report_dir)
    model = Pipeline(
        [
            ("scaler", scaler),
            ("pca", pca if pca is not None else "passthrough"),
            ("model", best_model),
        ]
    )
    joblib.dump(model, model_path)

    summary = {
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": history[best_epoch - 1]["val_macro_f1"],
        "best_val_loss": history[best_epoch - 1]["val_loss"],
        "class_names": list(class_names or (str(label) for label in classes)),
        "use_pca": use_pca,
        "model_path": str(model_path),
        "history_csv": str(report_dir / "epoch_history.csv"),
        "plot_path": str(report_dir / "epoch_training.png"),
    }
    (report_dir / "epoch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Huấn luyện SGD theo epoch và lưu biểu đồ theo dõi"
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-pca", action="store_true", help="Tắt PCA")
    parser.add_argument("--force-preprocess", action="store_true")
    args = parser.parse_args(argv)

    if args.epochs < 1:
        parser.error("--epochs phải lớn hơn 0.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate phải lớn hơn 0.")

    data_dir = args.data_dir or _default_data_dir()
    missing_splits = [
        split for split in ("train", "val", "test")
        if not (data_dir / split).is_dir()
    ]
    if missing_splits:
        print(
            f"LỖI: Thiếu split {', '.join(missing_splits)} trong {data_dir}. "
            "Hãy kiểm tra --data-dir hoặc chạy bước chuẩn bị dataset."
        )
        return 2

    try:
        data = _load_data(
            data_dir,
            cache_dir=args.cache_dir,
            force_preprocess=args.force_preprocess,
        )
        X_train, y_train, X_val, y_val = data[0], data[1], data[2], data[3]
        class_names = data[9]
        dataset_mode = data[-1]
        default_model_dir = Path(
            "models-whole-scene" if dataset_mode == "whole-scene" else "models"
        )
        model_dir = args.model_dir or default_model_dir
        reports_dir = args.reports_dir or model_dir.parent / (
            "reports-whole-scene" if dataset_mode == "whole-scene" else "reports"
        )
        run_id = _new_run_id()
        summary = train_epochs(
            X_train,
            y_train,
            X_val,
            y_val,
            report_dir=reports_dir / "epoch-runs" / run_id,
            model_path=model_dir / "epoch-runs" / run_id / "classifier.joblib",
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            use_pca=not args.no_pca,
            random_state=args.random_state,
            class_names=class_names,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"LỖI: {exc}")
        return 2

    print(f"Epoch tốt nhất: {summary['best_epoch']}/{args.epochs}")
    print(f"Biểu đồ: {summary['plot_path']}")
    print(f"Model: {summary['model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
