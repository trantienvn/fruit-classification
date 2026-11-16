"""
torch_classifier.py — Bộ phân loại PyTorch MLP có khả năng huấn luyện trên CUDA GPU.

Tương thích hoàn toàn với scikit-learn API (BaseEstimator, ClassifierMixin, Pipeline, GridSearchCV).
Hỗ trợ tự động phát hiện và huấn luyện trên NVIDIA CUDA GPU giúp tăng tốc độ gấp nhiều lần.
"""

from __future__ import annotations

import logging
import time
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
        self.device = device
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self._net = None
        self._resolved_device = None

    def _get_device(self):
        torch, _ = _check_torch()
        if self.device == "auto" or self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dev = torch.device(self.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            logger.warning("[TorchMLPClassifier] PyTorch không có CUDA, chuyển về CPU.")
            return torch.device("cpu")
        return dev

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPClassifier":
        """Huấn luyện mạng nơ-ron MLP trên GPU/CPU."""
        torch, nn = _check_torch()
        torch.manual_seed(self.random_state)

        dev = self._get_device()
        self._resolved_device = dev

        # Mã hóa nhãn dạng số nguyên liên tục
        classes, encoded_y = np.unique(y, return_inverse=True)
        self.classes_ = classes
        n_classes = len(classes)

        X_arr = np.asarray(X, dtype=np.float32)
        in_features = X_arr.shape[1]

        # Xây dựng kiến trúc mạng
        self._net = _MLPNet.build(
            in_features, self.hidden_dim1, self.hidden_dim2, n_classes, self.dropout
        ).to(dev)

        X_tensor = torch.as_tensor(X_arr, dtype=torch.float32)
        y_tensor = torch.as_tensor(encoded_y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )
        criterion = nn.CrossEntropyLoss()

        t0 = time.perf_counter()
        self._net.train()

        use_amp = dev.type == "cuda"
        for epoch in range(self.epochs):
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

            scheduler.step()

        elapsed = time.perf_counter() - t0
        device_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
        logger.debug(
            f"[TorchMLPClassifier] Huấn luyện {self.epochs} epochs trên {device_name} "
            f"hoàn tất trong {elapsed:.2f}s"
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
