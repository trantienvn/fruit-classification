"""FastAPI app, khởi tạo model một lần và phục vụ giao diện tĩnh."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.dataset import CLASS_NAMES
from src.predict import FruitClassifier

from app.web.config import Settings
from app.web.routes import health, history, model_info, prediction, statistics
from app.web.services.history_service import HistoryService

logger = logging.getLogger(__name__)
ClassifierFactory = Callable[[Path | str], FruitClassifier]
PAGE_NAMES = {"dashboard", "recognition", "history", "statistics", "model-info"}


def create_app(
    settings: Settings | None = None,
    classifier_factory: ClassifierFactory = FruitClassifier,
) -> FastAPI:
    """Tạo app có thể cấu hình để kiểm thử mà không cần model thật."""
    app_settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        history_service = HistoryService(app_settings.history_db)
        history_service.initialize()
        app.state.settings = app_settings
        app.state.history_service = history_service
        app.state.classifier = None
        app.state.class_names = list(CLASS_NAMES)
        try:
            classifier = classifier_factory(app_settings.model_dir)
            classifier.load()
            app.state.classifier = classifier
            app.state.class_names = classifier.get_info().get(
                "class_names", list(CLASS_NAMES)
            )
        except Exception as error:
            logger.warning("Model chưa sẵn sàng: %s", error)
            app.state.model_error = str(error)
        yield

    application = FastAPI(
        title="Vườn chín API",
        summary="API phân loại độ chín trái cây bằng model học máy hiện có.",
        description=(
            "Nhận ảnh JPG, PNG, BMP hoặc WEBP; dùng trực tiếp FruitClassifier. "
            "Lịch sử chỉ lưu metadata, không lưu ảnh tải lên."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = app_settings
    application.include_router(health.router, prefix="/api")
    application.include_router(prediction.router, prefix="/api")
    application.include_router(model_info.router, prefix="/api")
    application.include_router(history.router, prefix="/api")
    application.include_router(statistics.router, prefix="/api")

    @application.exception_handler(HTTPException)
    async def http_error_handler(_: Request, error: HTTPException) -> JSONResponse:
        if isinstance(error.detail, dict):
            code = str(error.detail.get("code", "request_error"))
            message = str(error.detail.get("message", "Request không hợp lệ."))
        else:
            code = "request_error"
            message = str(error.detail)
        return JSONResponse(
            status_code=error.status_code,
            content={"success": False, "error": {"code": code, "message": message}},
            headers=error.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _: Request, __: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "code": "validation_error",
                    "message": "Thông tin request không hợp lệ.",
                },
            },
        )

    frontend = app_settings.frontend_root
    static_dir = frontend / "static"
    application.mount(
        "/static", StaticFiles(directory=static_dir, check_dir=True), name="static"
    )
    if app_settings.reports_root.is_dir():
        application.mount(
            "/reports",
            StaticFiles(directory=app_settings.reports_root, check_dir=True),
            name="reports",
        )

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(frontend / "index.html")

    @application.get("/pages/{page_name}.html", include_in_schema=False)
    def page(page_name: str) -> FileResponse:
        if page_name not in PAGE_NAMES:
            raise HTTPException(status_code=404, detail="Không tìm thấy trang.")
        return FileResponse(frontend / "pages" / f"{page_name}.html")

    return application


app = create_app()
