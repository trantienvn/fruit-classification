"""Cấu hình đường dẫn và giới hạn cho ứng dụng web Deep Learning.
Tương thích hoàn toàn với môi trường Local và Vercel Serverless Functions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = PROJECT_ROOT / "frontend" if (PROJECT_ROOT / "frontend").is_dir() else (WEB_ROOT / "frontend")

IS_VERCEL = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB


@dataclass(frozen=True)
class Settings:
    """Các giá trị có thể cấu hình qua biến môi trường."""

    model_dir: Path = PROJECT_ROOT / "models"
    dl_model_dir: Path = PROJECT_ROOT / "models"
    history_db: Path = (Path("/tmp/history.sqlite3") if IS_VERCEL else WEB_ROOT / "data" / "history.sqlite3")
    frontend_root: Path = FRONTEND_ROOT
    reports_root: Path = PROJECT_ROOT / "reports"
    max_upload_bytes: int = MAX_UPLOAD_SIZE

    @classmethod
    def from_environment(cls) -> "Settings":
        dl_dir = Path(os.environ.get("FRUIT_DL_MODEL_DIR") or os.environ.get("FRUIT_MODEL_DIR") or (PROJECT_ROOT / "models"))
        reports_dir = os.environ.get("FRUIT_REPORTS_DIR")
        if reports_dir is None:
            if dl_dir.name.startswith("models-"):
                candidate_reports = dl_dir.parent / f"reports-{dl_dir.name.removeprefix('models-')}"
            else:
                candidate_reports = PROJECT_ROOT / "reports"
            reports_dir = candidate_reports

        if IS_VERCEL:
            default_history_db = Path("/tmp/history.sqlite3")
        else:
            default_history_db = WEB_ROOT / "data" / "history.sqlite3"

        return cls(
            model_dir=dl_dir,
            dl_model_dir=dl_dir,
            history_db=Path(os.environ.get("FRUIT_HISTORY_DB", default_history_db)),
            frontend_root=Path(os.environ.get("FRUIT_FRONTEND_DIR", FRONTEND_ROOT)),
            reports_root=Path(reports_dir),
            max_upload_bytes=int(
                os.environ.get("FRUIT_MAX_UPLOAD_BYTES", MAX_UPLOAD_SIZE)
            ),
        )
