"""Huấn luyện SGD theo epoch và lưu lịch sử từng lần chạy.

Chạy:
    python -m src.train_epochs --epochs 50
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from src.evaluate import epoch_report
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

    loss_figure, loss_plot_axis = plt.subplots(figsize=(8, 5))
    loss_plot_axis.plot(
        epochs, [row["train_loss"] for row in history], label="Train"
    )
    loss_plot_axis.plot(
        epochs, [row["val_loss"] for row in history], label="Validation"
    )
    loss_plot_axis.set(
        title="Xu hướng Loss theo Epoch",
        xlabel="Epoch",
        ylabel="Log loss",
    )
    loss_plot_axis.grid(alpha=0.25)
    loss_plot_axis.legend()
    loss_figure.tight_layout()
    loss_figure.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(loss_figure)


def _data_signature(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    joblib.dump(checkpoint, temporary_path)
    temporary_path.replace(path)


def _resolve_resume_run(reports_root: Path, run_id: str) -> str:
    runs_dir = reports_root / "epoch-runs"
    if run_id == "latest":
        available_runs = sorted(
            (
                path.name
                for path in runs_dir.iterdir()
                if path.is_dir() and (path / "epoch_checkpoint.joblib").is_file()
            ),
            reverse=True,
        ) if runs_dir.is_dir() else []
        if not available_runs:
            raise FileNotFoundError(f"Không tìm thấy checkpoint trong {runs_dir}.")
        return available_runs[0]

    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("RUN_ID phải là tên thư mục run, không phải đường dẫn.")
    if not (runs_dir / run_id / "epoch_checkpoint.joblib").is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint cho run '{run_id}' trong {runs_dir}."
        )
    return run_id


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
    resume: bool = False,
    test_data: tuple[np.ndarray, np.ndarray, list[np.ndarray]] | None = None,
) -> dict:
    """Huấn luyện thêm epoch, có thể tiếp tục từ checkpoint của cùng run."""
    X_train = np.asarray(X_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    classes = _validate_data(
        X_train, y_train, X_val, y_val, epochs, learning_rate
    )
    if class_names is not None and len(class_names) != len(classes):
        raise ValueError("Số class_names phải bằng số lớp trong tập train.")

    resolved_class_names = tuple(
        class_names if class_names is not None else (str(label) for label in classes)
    )
    config = {
        "learning_rate": learning_rate,
        "use_pca": use_pca,
        "random_state": random_state,
        "class_names": list(resolved_class_names),
    }
    data_signature = _data_signature(X_train, y_train, X_val, y_val)
    checkpoint_path = report_dir / "epoch_checkpoint.joblib"
    preprocessing_path = report_dir / "epoch_preprocessing.joblib"

    if resume:
        if not checkpoint_path.is_file() or not preprocessing_path.is_file():
            raise FileNotFoundError(
                f"Run này chưa có checkpoint đầy đủ: {checkpoint_path}."
            )
        checkpoint = joblib.load(checkpoint_path)
        required_fields = {
            "version", "config", "data_signature", "classes", "history",
            "latest_classifier", "best_classifier", "best_epoch",
        }
        if not isinstance(checkpoint, dict) or not required_fields.issubset(checkpoint):
            raise ValueError(f"Checkpoint không hợp lệ: {checkpoint_path}.")
        if checkpoint["version"] != 1:
            raise ValueError("Phiên bản checkpoint không được hỗ trợ.")
        if checkpoint["data_signature"] != data_signature:
            raise ValueError(
                "Dataset train/validation đã thay đổi; không thể resume run này."
            )
        if checkpoint["config"] != config:
            raise ValueError(
                "Cấu hình learning rate, PCA, random state hoặc class names khác "
                "checkpoint; hãy khôi phục cấu hình cũ hoặc tạo run mới."
            )
        if not np.array_equal(checkpoint["classes"], classes):
            raise ValueError("Danh sách lớp không khớp checkpoint.")

        preprocessing = joblib.load(preprocessing_path)
        if (
            not isinstance(preprocessing, dict)
            or "scaler" not in preprocessing
            or "pca" not in preprocessing
        ):
            raise ValueError(f"Tiền xử lý trong run không hợp lệ: {preprocessing_path}.")
        scaler = preprocessing["scaler"]
        pca = preprocessing["pca"]
        if not isinstance(scaler, StandardScaler) or (
            pca is not None and not isinstance(pca, PCA)
        ):
            raise ValueError(f"Tiền xử lý trong run không hợp lệ: {preprocessing_path}.")
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        if pca is not None:
            X_train_scaled = pca.transform(X_train_scaled)
            X_val_scaled = pca.transform(X_val_scaled)

        classifier = checkpoint["latest_classifier"]
        best_model = checkpoint["best_classifier"]
        history = checkpoint["history"]
        best_epoch = int(checkpoint["best_epoch"])
        if not isinstance(classifier, SGDClassifier) or not isinstance(
            best_model, SGDClassifier
        ):
            raise ValueError(f"Model trong checkpoint không hợp lệ: {checkpoint_path}.")
        history_fields = {
            "epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy",
            "train_macro_f1", "val_macro_f1",
        }
        if (
            not isinstance(history, list)
            or not history
            or any(
                not isinstance(row, dict) or not history_fields.issubset(row)
                for row in history
            )
            or [row["epoch"] for row in history] != list(range(1, len(history) + 1))
            or best_epoch < 1
            or best_epoch > len(history)
        ):
            raise ValueError(f"Lịch sử epoch trong checkpoint không hợp lệ: {checkpoint_path}.")
        start_epoch = len(history)
        previous_best = history[best_epoch - 1]
        best_score = (
            float(previous_best["val_macro_f1"]),
            -float(previous_best["val_loss"]),
        )
    else:
        if checkpoint_path.exists():
            raise FileExistsError(
                f"Run đã có checkpoint tại {checkpoint_path}; dùng resume=True "
                "hoặc chọn thư mục run mới."
            )
        report_dir.mkdir(parents=True, exist_ok=True)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        pca: PCA | None = None
        if use_pca:
            pca = PCA(n_components=0.95, random_state=random_state)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_val_scaled = pca.transform(X_val_scaled)
        joblib.dump({"scaler": scaler, "pca": pca}, preprocessing_path)

        classifier = SGDClassifier(
            loss="log_loss",
            learning_rate="constant",
            eta0=learning_rate,
            shuffle=False,
            random_state=random_state,
        )
        history = []
        best_model = None
        best_epoch = 0
        best_score = (-1.0, float("-inf"))
        start_epoch = 0

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        rng = np.random.default_rng(random_state + epoch - 1)
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
        _save_checkpoint(
            checkpoint_path,
            {
                "version": 1,
                "config": config,
                "data_signature": data_signature,
                "classes": classes,
                "history": history,
                "latest_classifier": classifier,
                "best_classifier": best_model,
                "best_epoch": best_epoch,
            },
        )

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

    test_metrics = None
    if test_data is not None:
        X_test, y_test, test_images = test_data
        X_test = np.asarray(X_test, dtype=np.float64)
        y_test = np.asarray(y_test)
        if (
            X_test.ndim != 2
            or X_test.shape[1] != X_train.shape[1]
            or len(X_test) != len(y_test)
            or not np.isin(y_test, classes).all()
        ):
            raise ValueError("Dữ liệu test không khớp với train.")
        best_epoch_metrics = history[best_epoch - 1]
        test_metrics = epoch_report(
            model,
            X_test,
            y_test,
            resolved_class_names,
            y_train,
            y_val,
            test_images,
            {
                "accuracy": best_epoch_metrics["val_accuracy"],
                "macro_f1": best_epoch_metrics["val_macro_f1"],
            },
            report_dir,
        )

    summary = {
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "resumed": resume,
        "best_epoch": best_epoch,
        "best_val_macro_f1": history[best_epoch - 1]["val_macro_f1"],
        "best_val_loss": history[best_epoch - 1]["val_loss"],
        "class_names": list(resolved_class_names),
        "use_pca": use_pca,
        "test_metrics": test_metrics,
        "model_path": str(model_path),
        "checkpoint_path": str(checkpoint_path),
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
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        metavar="RUN_ID",
        help="Tiếp tục run đã lưu; bỏ RUN_ID để tiếp tục run mới nhất",
    )
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
        X_test, y_test, test_images = data[4], data[5], data[7]
        class_names = data[9]
        dataset_mode = data[-1]
        default_model_dir = Path(
            "models-whole-scene" if dataset_mode == "whole-scene" else "models"
        )
        model_dir = args.model_dir or default_model_dir
        reports_dir = args.reports_dir or model_dir.parent / (
            "reports-whole-scene" if dataset_mode == "whole-scene" else "reports"
        )
        if args.resume is not None:
            run_id = _resolve_resume_run(reports_dir, args.resume)
        else:
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
            resume=args.resume is not None,
            test_data=(X_test, y_test, test_images),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"LỖI: {exc}")
        return 2

    run_mode = "tiếp tục checkpoint" if summary["resumed"] else "run mới"
    print(f"Epoch tốt nhất: {summary['best_epoch']}/{summary['epochs_completed']}")
    print(f"Đã chạy {args.epochs} epoch trong {run_mode}.")
    print(f"Biểu đồ: {summary['plot_path']}")
    print(f"Model: {summary['model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
