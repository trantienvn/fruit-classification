"""Tiện ích kiểm tra dataset và tiền xử lý ảnh (khung ban đầu)."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(directory: str | Path) -> list[Path]:
    """Liệt kê ảnh hỗ trợ trong thư mục, theo thứ tự ổn định."""
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_image(path: str | Path) -> bool:
    """Kiểm tra ảnh có thể được mở và giải mã bởi Pillow."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False

