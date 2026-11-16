"""Entry point cho Vercel Serverless Functions.
Vercel tự động nhận diện file này và khởi tạo ứng dụng FastAPI qua ASGI.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Thêm thư mục gốc của dự án vào sys.path để import các module app, src_dl, src_tabular
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.web.main import app

# Export app cho Vercel ASGI runner
__all__ = ["app"]
