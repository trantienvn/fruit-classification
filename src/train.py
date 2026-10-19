"""Điểm vào huấn luyện; chưa triển khai cho đến khi có dataset thật."""

from pathlib import Path

from src.data import find_images


def require_dataset(data_dir: str | Path = "data/raw") -> None:
    """Báo thiếu dữ liệu một cách tường minh, không tạo kết quả giả."""
    root = Path(data_dir)
    if not find_images(root):
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu trong {root}. Hãy chuẩn bị dữ liệu theo data/README.md."
        )


if __name__ == "__main__":
    require_dataset()
    raise NotImplementedError("Pipeline huấn luyện sẽ được bổ sung sau khi xác nhận định dạng dữ liệu.")
