"""Entry point cho Vercel Serverless Functions.
Vercel tự động nhận diện file này và khởi tạo ứng dụng FastAPI qua ASGI.
Hỗ trợ tự động chuẩn hóa URL path khi Vercel thực hiện rewrite /api/index.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Thêm thư mục gốc của dự án vào sys.path để import các module app, src
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.web.main import app as fastapi_app


class VercelPathMiddleware:
    """Tự động khôi phục đúng đường dẫn request khi Vercel rewrite đến /api/index."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            # Header x-matched-path do Vercel tự động gán, chứa URL gốc người dùng truy cập
            matched_path = headers.get(b"x-matched-path")
            raw_path = scope.get("path", "/")

            if matched_path:
                decoded_path = matched_path.decode("utf-8")
                # Nếu matched_path là /api/index, chuẩn hóa về /
                if decoded_path in ("/api/index", "/api/index.py", "/api"):
                    scope["path"] = "/"
                else:
                    scope["path"] = decoded_path
            elif raw_path in ("/api/index", "/api/index.py", "/api"):
                scope["path"] = "/"

        await self.app(scope, receive, send)


app = VercelPathMiddleware(fastapi_app)

__all__ = ["app"]
