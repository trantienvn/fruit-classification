"""
torch_classifier.py — Bộ phân loại PyTorch MLP có khả năng huấn luyện trên CUDA GPU,
theo dõi xu hướng Loss/Accuracy qua từng Epoch và hỗ trợ tiếp tục huấn luyện (resume/warm_start).

Tương thích hoàn toàn với scikit-learn API (BaseEstimator, ClassifierMixin, Pipeline, GridSearchCV).
Hỗ trợ tự động phát hiện và huấn luyện trên NVIDIA CUDA GPU giúp tăng tốc độ gấp nhiều lần.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


def _check_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as e:
        raise RuntimeError(
            "Cần cài đặt PyTorch để sử dụng TorchMLPClassifier:\n"
            "    pip install torch torchvision"
        ) from e


class _MLPNet:
    """Xây dựng mạng nơ-ron MLP sâu với BatchNorm và Dropout."""

    @staticmethod
    def build(in_features: int, hidden_dim1: int, hidden_dim2: int, n_classes: int, dropout: float = 0.25):
        torch, nn = _check_torch()
        return nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden_dim2, n_classes),
        )


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Bộ phân loại mạng nơ-ron đa tầng (MLP) huấn luyện trên CUDA GPU.

    Tham số:
        hidden_dim1  : số nơ-ron tầng ẩn 1 (mặc định: 256).
        hidden_dim2  : số nơ-ron tầng ẩn 2 (mặc định: 128).
        lr           : tốc độ học (learning rate, mặc định: 1e-3).
        weight_decay : hệ số điều chuẩn L2 cho AdamW (mặc định: 1e-4).
        epochs       : số vòng lặp huấn luyện (mặc định: 60).
        batch_size   : kích thước mini-batch (mặc định: 64).
        dropout      : xác suất ngắt kết nối (mặc định: 0.25).
        warm_start   : tiếp tục huấn luyện từ trọng số hiện có nếu True (mặc định: False).
        device       : "auto" (tự nhận diện CUDA) | "cuda" | "cpu".
        random_state : hạt giống ngẫu nhiên (mặc định: 42).
    """

    def __init__(
        self,
        hidden_dim1: int = 256,
        hidden_dim2: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 60,
        batch_size: int = 64,
        dropout: float = 0.25,
        warm_start: bool = False,
        device: str = "auto",
        random_state: int = 42,
    ) -> None:
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.warm_start = warm_start
        self.device = device
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self._net = None
        self._resolved_device = None
        self.history_: list[dict[str, Any]] = []

    def _get_device(self):
        torch, _ = _check_torch()
        if self.device == "auto" or self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dev = torch.device(self.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            logger.warning("[TorchMLPClassifier] PyTorch không có CUDA, chuyển về CPU.")
            return torch.device("cpu")
        return dev

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "TorchMLPClassifier":
        """
        Huấn luyện mạng nơ-ron MLP trên GPU/CPU, ghi lại lịch sử Loss theo từng Epoch.

        Hỗ trợ warm_start=True để tiếp tục huấn luyện nối tiếp số epoch trước đó.
        """
        torch, nn = _check_torch()
        torch.manual_seed(self.random_state)

        dev = self._get_device()
        self._resolved_device = dev

        # Xác định nhãn lớp
        if self.classes_ is None or not self.warm_start:
            classes, encoded_y = np.unique(y, return_inverse=True)
            self.classes_ = classes
        else:
            classes = self.classes_
            class_to_idx = {c: i for i, c in enumerate(classes)}
            encoded_y = np.array([class_to_idx.get(yi, 0) for yi in y], dtype=np.int64)

        n_classes = len(classes)
        X_arr = np.asarray(X, dtype=np.float32)
        in_features = X_arr.shape[1]

        # Khởi tạo mạng hoặc tiếp tục từ trọng số đã có
        start_epoch = 0
        if self.warm_start and self._net is not None:
            self._net = self._net.to(dev)
            if self.history_:
                start_epoch = self.history_[-1]["epoch"]
            logger.info(
                f"[TorchMLPClassifier] Tiếp tục huấn luyện từ epoch {start_epoch} "
                f"thêm {self.epochs} epochs..."
            )
        else:
            self._net = _MLPNet.build(
                in_features, self.hidden_dim1, self.hidden_dim2, n_classes, self.dropout
            ).to(dev)
            self.history_ = []

        X_tensor = torch.as_tensor(X_arr, dtype=torch.float32)
        y_tensor = torch.as_tensor(encoded_y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        # Xử lý tập validation nếu được truyền
        has_val = eval_set is not None
        if has_val:
            X_val_raw, y_val_raw = eval_set
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y_val_enc = np.array([class_to_idx.get(yi, 0) for yi in y_val_raw], dtype=np.int64)
            val_X_tensor = torch.as_tensor(np.asarray(X_val_raw, dtype=np.float32), dtype=torch.float32).to(dev)
            val_y_tensor = torch.as_tensor(y_val_enc, dtype=torch.long).to(dev)

        optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )
        criterion = nn.CrossEntropyLoss()

        t0 = time.perf_counter()
        use_amp = dev.type == "cuda"

        total_epochs = start_epoch + self.epochs
        for current_idx in range(self.epochs):
            epoch_num = start_epoch + current_idx + 1
            self._net.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(dev, non_blocking=True)
                batch_y = batch_y.to(dev, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        logits = self._net(batch_x)
                        loss = criterion(logits, batch_y)
                else:
                    logits = self._net(batch_x)
                    loss = criterion(logits, batch_y)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct_train += (preds == batch_y).sum().item()
                total_train += batch_x.size(0)

            scheduler.step()

            train_loss = running_loss / max(1, total_train)
            train_acc = correct_train / max(1, total_train)

            # Đánh giá trên Validation
            val_loss = train_loss
            val_acc = train_acc
            if has_val:
                self._net.eval()
                with torch.inference_mode():
                    if use_amp:
                        with torch.amp.autocast(device_type="cuda"):
                            val_logits = self._net(val_X_tensor)
                            val_loss_t = criterion(val_logits, val_y_tensor)
                    else:
                        val_logits = self._net(val_X_tensor)
                        val_loss_t = criterion(val_logits, val_y_tensor)

                    val_loss = float(val_loss_t.item())
                    val_preds = torch.argmax(val_logits, dim=1)
                    val_acc = float((val_preds == val_y_tensor).sum().item() / max(1, len(val_y_tensor)))

            current_lr = float(optimizer.param_groups[0]["lr"])
            self.history_.append({
                "epoch": epoch_num,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "train_acc": round(train_acc, 5),
                "val_acc": round(val_acc, 5),
                "lr": round(current_lr, 6),
            })

            if epoch_num % 10 == 0 or epoch_num == total_epochs or epoch_num == 1:
                logger.info(
                    f"  Epoch [{epoch_num:3d}/{total_epochs:3d}] "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
                    f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
                )

        elapsed = time.perf_counter() - t0
        device_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
        logger.info(
            f"[TorchMLPClassifier] Huấn luyện {self.epochs} epochs trên {device_name} "
            f"hoàn tất trong {elapsed:.2f}s (Tổng epoch hiện tại: {total_epochs})"
        )

        self._net.eval()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán xác suất phân loại (softmax)."""
        torch, _ = _check_torch()
        if self._net is None:
            raise RuntimeError("Mô hình TorchMLPClassifier chưa được huấn luyện.")

        dev = self._resolved_device or self._get_device()
        self._net.eval()

        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.as_tensor(X_arr, dtype=torch.float32).to(dev)

        with torch.inference_mode():
            if dev.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = self._net(X_tensor)
            else:
                logits = self._net(X_tensor)
            probs = torch.softmax(logits, dim=-1)

        return probs.cpu().numpy().astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán nhãn lớp."""
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=-1)
        if self.classes_ is not None:
            return self.classes_[pred_indices]
        return pred_indices

    def __getstate__(self) -> dict[str, Any]:
        """Tự động chuyển weights về CPU trước khi pickle/joblib để an toàn."""
        state = self.__dict__.copy()
        if self._net is not None:
            state["_net_state_dict"] = {k: v.cpu() for k, v in self._net.state_dict().items()}
            state["_net"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Khôi phục mô hình và tự động đưa lên device phù hợp khi unpickle."""
        net_state = state.pop("_net_state_dict", None)
        self.__dict__.update(state)
        if not hasattr(self, "history_") or self.history_ is None:
            self.history_ = []
        if not hasattr(self, "warm_start"):
            self.warm_start = False
        if net_state is not None and self.classes_ is not None:
            dev = self._get_device()
            self._resolved_device = dev
            first_weight = next(iter(net_state.values()))
            in_features = first_weight.shape[1] if hasattr(first_weight, "shape") else 576
            self._net = _MLPNet.build(
                in_features, self.hidden_dim1, self.hidden_dim2, len(self.classes_), self.dropout
            )
            self._net.load_state_dict(net_state)
            self._net.to(dev)
            self._net.eval()


# ---------------------------------------------------------------------------
# Tiện ích trực quan hóa: Vẽ sơ đồ xu hướng Loss và Accuracy theo Epoch
# ---------------------------------------------------------------------------

def plot_loss_curve(
    history: list[dict[str, Any]],
    output_dir: Path | str,
    title_suffix: str = "",
) -> tuple[Path, Path]:
    """
    Vẽ và lưu sơ đồ xu hướng Loss và biểu đồ huấn luyện chi tiết theo Epoch.

    Tạo ra 2 file:
        1. loss_curve.png: Biểu đồ tập trung vào hàm mất mát (Loss curve) theo epoch.
        2. epoch_training.png: Biểu đồ kết hợp Loss và Accuracy/F1 qua các epoch.
        3. epoch_history.csv: Bảng số liệu chi tiết qua từng epoch.
        4. epoch_summary.json: Tóm tắt thông số tốt nhất.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not history:
        logger.warning("[plot_loss_curve] Lịch sử huấn luyện rỗng, không thể vẽ biểu đồ.")
        return output_dir / "loss_curve.png", output_dir / "epoch_training.png"

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row.get("train_acc", 0.0) for row in history]
    val_acc = [row.get("val_acc", 0.0) for row in history]

    # Tìm điểm tối ưu
    best_val_idx = int(np.argmin(val_loss))
    best_val_epoch = epochs[best_val_idx]
    min_val_loss = val_loss[best_val_idx]

    best_acc_idx = int(np.argmax(val_acc))
    best_acc_epoch = epochs[best_acc_idx]
    max_val_acc = val_acc[best_acc_idx]

    # 1. Vẽ file loss_curve.png (Biểu đồ xu hướng Loss theo Epoch)
    fig_loss, ax_loss = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax_loss.plot(epochs, train_loss, label="Train Loss (Tập huấn luyện)", color="#2d6a4f", linewidth=2.2, alpha=0.9)
    ax_loss.plot(epochs, val_loss, label="Validation Loss (Tập kiểm định)", color="#c9852d", linewidth=2.2, alpha=0.9)
    ax_loss.fill_between(epochs, train_loss, val_loss, color="#2d6a4f", alpha=0.08)

    # Đánh dấu điểm loss nhỏ nhất
    ax_loss.scatter([best_val_epoch], [min_val_loss], color="#a84d42", s=70, zorder=5)
    ax_loss.annotate(
        f"Min Val Loss: {min_val_loss:.4f}\n(Epoch {best_val_epoch})",
        xy=(best_val_epoch, min_val_loss),
        xytext=(best_val_epoch + max(1, len(epochs) // 15), min_val_loss + 0.05),
        arrowprops=dict(facecolor="#a84d42", shrink=0.08, width=1, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#a84d42",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff", ec="#a84d42", alpha=0.8),
    )

    title = "Sơ đồ xu hướng Loss theo Epoch"
    if title_suffix:
        title += f" — {title_suffix}"
    ax_loss.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax_loss.set_xlabel("Epoch", fontsize=11, fontweight="600")
    ax_loss.set_ylabel("Hàm mất mát (Cross-Entropy Loss)", fontsize=11, fontweight="600")
    ax_loss.grid(True, linestyle="--", alpha=0.35)
    ax_loss.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e7df")
    fig_loss.tight_layout()

    loss_curve_path = output_dir / "loss_curve.png"
    fig_loss.savefig(loss_curve_path)
    plt.close(fig_loss)

    # 2. Vẽ file epoch_training.png (Subplots: Loss và Accuracy)
    fig_all, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Trục 1: Loss
    ax1.plot(epochs, train_loss, label="Train", color="#2d6a4f", linewidth=2)
    ax1.plot(epochs, val_loss, label="Validation", color="#c9852d", linewidth=2)
    ax1.scatter([best_val_epoch], [min_val_loss], color="#a84d42", s=50, zorder=5)
    ax1.set_title("Cross-Entropy Loss theo Epoch", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()

    # Trục 2: Accuracy
    ax2.plot(epochs, [a * 100 for a in train_acc], label="Train Acc", color="#2d6a4f", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in val_acc], label="Val Acc", color="#3b82f6", linewidth=2)
    ax2.scatter([best_acc_epoch], [max_val_acc * 100], color="#2563eb", s=50, zorder=5)
    ax2.set_title("Độ chính xác (Accuracy %) theo Epoch", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Độ chính xác (%)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()

    fig_all.suptitle(f"Tiến trình huấn luyện mô hình sâu ({len(epochs)} Epochs)", fontsize=13, fontweight="bold")
    fig_all.tight_layout()

    epoch_training_path = output_dir / "epoch_training.png"
    fig_all.savefig(epoch_training_path)
    plt.close(fig_all)

    # 3. Lưu epoch_history.csv
    csv_path = output_dir / "epoch_history.csv"
    fieldnames = list(history[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    # 4. Lưu epoch_summary.json
    summary = {
        "total_epochs": len(epochs),
        "best_epoch": best_val_epoch,
        "min_val_loss": min_val_loss,
        "best_acc_epoch": best_acc_epoch,
        "max_val_acc": max_val_acc,
        "final_train_loss": train_loss[-1],
        "final_val_loss": val_loss[-1],
        "final_train_acc": train_acc[-1],
        "final_val_acc": val_acc[-1],
    }
    with open(output_dir / "epoch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[plot_loss_curve] Đã xuất sơ đồ xu hướng Loss: {loss_curve_path} "
        f"và biểu đồ huấn luyện: {epoch_training_path}"
    )

    return loss_curve_path, epoch_training_path
