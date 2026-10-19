"""Tiện ích dự đoán; suy luận mô hình sẽ được bổ sung khi có checkpoint."""

from pathlib import Path


def checkpoint_available(path: str | Path = "models/fruit_classifier.pth") -> bool:
    """Trả về True khi checkpoint được chỉ định đã tồn tại."""
    return Path(path).is_file()

