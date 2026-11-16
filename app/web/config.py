"""Cấu hình đường dẫn và giới hạn cho ứng dụng web (hỗ trợ cả môi trường Local và Vercel Serverless)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.predict import MAX_FILE_SIZE_BYTES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = WEB_ROOT / "frontend"

IS_VERCEL = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))


@dataclass(frozen=True)
class Settings:
    """Các giá trị có thể cấu hình qua biến môi trường."""

    model_dir: Path = PROJECT_ROOT / "models_dl"
    dl_model_dir: Path = PROJECT_ROOT / "models_dl"
    tabular_model_dir: Path = PROJECT_ROOT / "models_tabular"
    legacy_model_dir: Path = PROJECT_ROOT / "models"
    history_db: Path = (Path("/tmp/history.sqlite3") if IS_VERCEL else WEB_ROOT / "data" / "history.sqlite3")
    frontend_root: Path = FRONTEND_ROOT
    reports_root: Path = PROJECT_ROOT / "reports"
    reports_dl_root: Path = PROJECT_ROOT / "reports_dl"
    reports_tabular_root: Path = PROJECT_ROOT / "reports_tabular"
    max_upload_bytes: int = MAX_FILE_SIZE_BYTES

    @classmethod
    def from_environment(cls) -> "Settings":
        dl_dir = Path(os.environ.get("FRUIT_DL_MODEL_DIR", PROJECT_ROOT / "models_dl"))
        tabular_dir = Path(os.environ.get("FRUIT_TABULAR_MODEL_DIR", PROJECT_ROOT / "models_tabular"))
        legacy_dir = Path(os.environ.get("FRUIT_MODEL_DIR", PROJECT_ROOT / "models"))

        # Ưu tiên models_dl nếu có mô hình, ngược lại fallback sang legacy
        default_model_dir = dl_dir if (dl_dir / "classifier_dl.joblib").exists() else legacy_dir

        reports_dir = os.environ.get("FRUIT_REPORTS_DIR")
        if reports_dir is None:
            reports_dir = PROJECT_ROOT / "reports"

        # Trên Vercel, filesystem là read-only, chỉ /tmp có quyền ghi
        if IS_VERCEL:
            default_history_db = Path("/tmp/history.sqlite3")
        else:
            default_history_db = WEB_ROOT / "data" / "history.sqlite3"

        return cls(
            model_dir=default_model_dir,
            dl_model_dir=dl_dir,
            tabular_model_dir=tabular_dir,
            legacy_model_dir=legacy_dir,
            history_db=Path(os.environ.get("FRUIT_HISTORY_DB", default_history_db)),
            frontend_root=Path(os.environ.get("FRUIT_FRONTEND_DIR", FRONTEND_ROOT)),
            reports_root=Path(reports_dir),
            reports_dl_root=PROJECT_ROOT / "reports_dl",
            reports_tabular_root=PROJECT_ROOT / "reports_tabular",
            max_upload_bytes=int(
                os.environ.get("FRUIT_MAX_UPLOAD_BYTES", MAX_FILE_SIZE_BYTES)
            ),
        )
