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

    model_dir: Path = PROJECT_ROOT / "models-whole-scene"
    history_db: Path = WEB_ROOT / "data" / "history.sqlite3"
    frontend_root: Path = FRONTEND_ROOT
    reports_root: Path = PROJECT_ROOT / "reports-whole-scene"
    max_upload_bytes: int = MAX_FILE_SIZE_BYTES

    @classmethod
    def from_environment(cls) -> "Settings":
        model_dir = Path(
            os.environ.get("FRUIT_MODEL_DIR", PROJECT_ROOT / "models-whole-scene")
        )
        reports_dir = os.environ.get("FRUIT_REPORTS_DIR")
        if reports_dir is None:
            reports_dir = model_dir.parent / "reports-whole-scene"

        return cls(
            model_dir=model_dir,
            history_db=Path(
                os.environ.get(
                    "FRUIT_HISTORY_DB", WEB_ROOT / "data" / "history.sqlite3"
                )
            ),
            frontend_root=Path(os.environ.get("FRUIT_FRONTEND_DIR", FRONTEND_ROOT)),
            reports_root=Path(reports_dir),
            max_upload_bytes=int(
                os.environ.get("FRUIT_MAX_UPLOAD_BYTES", MAX_FILE_SIZE_BYTES)
            ),
        )
