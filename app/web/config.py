"""Cấu hình đường dẫn và giới hạn cho ứng dụng web."""

from __future__ import annotations

import json
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
        model_dir = Path(
            os.environ.get("FRUIT_MODEL_DIR", PROJECT_ROOT / "models")
        )
        reports_dir = os.environ.get("FRUIT_REPORTS_DIR")
        if reports_dir is None:
            try:
                model_config = json.loads(
                    (model_dir / "config.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                model_config = {}
            is_whole_scene = (
                model_dir.name == "models-whole-scene"
                or (
                    isinstance(model_config, dict)
                    and model_config.get("dataset_mode") == "whole-scene"
                )
            )
            report_folder = "reports-whole-scene" if is_whole_scene else "reports"
            reports_dir = model_dir.parent / report_folder

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
