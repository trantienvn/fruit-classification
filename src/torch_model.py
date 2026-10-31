"""Bộ phân loại MLP PyTorch tùy chọn, có thể huấn luyện trên CUDA.

Mô-đun không import PyTorch ở cấp module để pipeline scikit-learn mặc định
không cần cài dependency này.
"""

from __future__ import annotations

import numpy as np


def _import_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "Chế độ GPU cần PyTorch. Cài bản CUDA phù hợp với máy theo "
            "https://pytorch.org/get-started/locally/ rồi chạy lại."
        ) from exc
    return torch, nn


def ensure_cuda_available() -> str:
    """Kiểm tra CUDA và trả về tên GPU đầu tiên."""
    torch, _ = _import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch không tìm thấy CUDA. Hãy kiểm tra GPU NVIDIA, driver và "
            "cài đúng bản PyTorch CUDA. Có thể kiểm tra bằng "
            "`torch.cuda.is_available()`."
        )
    return torch.cuda.get_device_name(0)


class TorchMLPClassifier:
    """MLP cho vector đặc trưng, giao diện predict tương thích pipeline hiện có.

    Trọng số được lưu dưới dạng NumPy để artifact không phụ thuộc thiết bị CUDA
    đã dùng lúc train. Khi dự đoán, model dùng CUDA nếu khả dụng, nếu không thì
    chạy trên CPU.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        hidden_sizes: tuple[int, ...] = (128, 64),
        patience: int = 12,
        random_state: int = 42,
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_sizes = hidden_sizes
        self.patience = patience
        self.random_state = random_state
        self._runtime_model = None
        self._runtime_device = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_runtime_model"] = None
        state["_runtime_device"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._runtime_model = None
        self._runtime_device = None

    @staticmethod
    def _build_network(nn, n_features: int, n_classes: int, hidden_sizes):
        layers = []
        input_size = n_features
        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(p=0.2)]
            )
            input_size = hidden_size
        layers.append(nn.Linear(input_size, n_classes))
        return nn.Sequential(*layers)

    def fit(self, X, y, *, validation_data=None):
        """Huấn luyện MLP; validation tùy chọn dùng cho early stopping."""
        torch, nn = _import_torch()
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device phải là 'cpu' hoặc 'cuda'.")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Không có CUDA khả dụng. Cài PyTorch CUDA phù hợp hoặc dùng "
                "`--device cpu`."
            )
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size và patience phải lớn hơn 0.")
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes phải chứa các số nguyên dương.")

        target_device = torch.device(self.device)
        X_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y)
        if X_array.ndim != 2 or len(X_array) != len(y_array) or not len(y_array):
            raise ValueError("X phải là ma trận không rỗng, cùng số dòng với y.")

        self.classes_, encoded_y = np.unique(y_array, return_inverse=True)
        self.n_features_in_ = int(X_array.shape[1])
        self.n_classes_ = int(len(self.classes_))

        torch.manual_seed(self.random_state)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.random_state)

        network = self._build_network(
            nn, self.n_features_in_, self.n_classes_, self.hidden_sizes
        ).to(target_device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        X_tensor = torch.as_tensor(X_array, dtype=torch.float32, device=target_device)
        y_tensor = torch.as_tensor(
            encoded_y, dtype=torch.long, device=target_device
        )

        X_val_tensor = y_val_tensor = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_array = np.asarray(X_val, dtype=np.float32)
            y_val_array = np.asarray(y_val)
            encoded_val = np.searchsorted(self.classes_, y_val_array)
            if np.any(encoded_val >= self.n_classes_) or not np.array_equal(
                self.classes_[encoded_val], y_val_array
            ):
                raise ValueError("y validation chứa lớp không có trong y train.")
            if X_val_array.ndim != 2 or X_val_array.shape[1] != self.n_features_in_:
                raise ValueError("X validation phải có cùng số đặc trưng với X train.")
            X_val_tensor = torch.as_tensor(
                X_val_array, dtype=torch.float32, device=target_device
            )
            y_val_tensor = torch.as_tensor(
                encoded_val, dtype=torch.long, device=target_device
            )

        best_state = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        self.best_epoch_ = self.epochs

        for epoch in range(self.epochs):
            network.train()
            order = torch.randperm(len(y_tensor), device=target_device)
            for indices in order.split(self.batch_size):
                optimizer.zero_grad(set_to_none=True)
                logits = network(X_tensor[indices])
                loss = loss_fn(logits, y_tensor[indices])
                loss.backward()
                optimizer.step()

            if X_val_tensor is not None and y_val_tensor is not None:
                network.eval()
                with torch.inference_mode():
                    val_loss = float(
                        loss_fn(network(X_val_tensor), y_val_tensor).item()
                    )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in network.state_dict().items()
                    }
                    self.best_epoch_ = epoch + 1
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        break

        if best_state is not None:
            network.load_state_dict(best_state)
        self.n_epochs_trained_ = epoch + 1
        self.state_ = {
            name: value.detach().cpu().numpy().copy()
            for name, value in network.state_dict().items()
        }
        self._runtime_model = None
        self._runtime_device = None
        return self

    def _get_runtime_model(self):
        torch, nn = _import_torch()
        if not hasattr(self, "state_"):
            raise RuntimeError("Model chưa được huấn luyện.")

        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        target_device = torch.device("cuda" if use_cuda else "cpu")
        if self._runtime_model is None or self._runtime_device != str(target_device):
            network = self._build_network(
                nn,
                self.n_features_in_,
                self.n_classes_,
                self.hidden_sizes,
            ).to(target_device)
            state = {
                name: torch.as_tensor(value, dtype=torch.float32, device=target_device)
                for name, value in self.state_.items()
            }
            network.load_state_dict(state)
            network.eval()
            self._runtime_model = network
            self._runtime_device = str(target_device)
        return torch, self._runtime_model, target_device

    def predict_proba(self, X) -> np.ndarray:
        if not hasattr(self, "state_"):
            raise RuntimeError("Model chưa được huấn luyện.")
        X_array = np.asarray(X, dtype=np.float32)
        if X_array.ndim != 2 or X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X phải có dạng (n_samples, {self.n_features_in_})."
            )
        torch, network, target_device = self._get_runtime_model()
        X_tensor = torch.as_tensor(
            X_array, dtype=torch.float32, device=target_device
        )
        with torch.inference_mode():
            probabilities = torch.softmax(network(X_tensor), dim=1)
        return probabilities.cpu().numpy()

    def predict(self, X) -> np.ndarray:
        encoded = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[encoded]
