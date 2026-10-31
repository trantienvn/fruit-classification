"""Cấu hình đường dẫn và giới hạn cho ứng dụng web."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.predict import MAX_FILE_SIZE_BYTES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = WEB_ROOT / "frontend"


@dataclass(frozen=True)
class Settings:
    """Các giá trị có thể cấu hình qua biến môi trường."""

    model_dir: Path = PROJECT_ROOT / "models"
    history_db: Path = WEB_ROOT / "data" / "history.sqlite3"
    frontend_root: Path = FRONTEND_ROOT
    reports_root: Path = PROJECT_ROOT / "reports"
    max_upload_bytes: int = MAX_FILE_SIZE_BYTES

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            model_dir=Path(os.environ.get("FRUIT_MODEL_DIR", PROJECT_ROOT / "models")),
            history_db=Path(
                os.environ.get(
                    "FRUIT_HISTORY_DB", WEB_ROOT / "data" / "history.sqlite3"
                )
            ),
            frontend_root=Path(os.environ.get("FRUIT_FRONTEND_DIR", FRONTEND_ROOT)),
            reports_root=Path(os.environ.get("FRUIT_REPORTS_DIR", PROJECT_ROOT / "reports")),
            max_upload_bytes=int(
                os.environ.get("FRUIT_MAX_UPLOAD_BYTES", MAX_FILE_SIZE_BYTES)
            ),
        )
