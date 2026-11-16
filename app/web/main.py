"""FastAPI app phục vụ mô hình Deep Learning phân loại ảnh trái cây và giao diện web hiện đại.
Tương thích hoàn toàn với môi trường Local và Vercel Serverless Functions.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.web.config import Settings
from app.web.labels import FULL_CLASS_MAP_VI
from app.web.routes import (
    health,
    history,
    model_info,
    prediction,
    statistics,
)
from app.web.services.history_service import HistoryService

logger = logging.getLogger(__name__)
ClassifierFactory = Callable[[Path | str], Any]
PAGE_NAMES = {"dashboard", "recognition", "history", "statistics", "model-info"}

DEFAULT_CLASSES = list(FULL_CLASS_MAP_VI.keys())


def create_app(
    settings: Settings | None = None,
    classifier_factory: ClassifierFactory | None = None,
) -> FastAPI:
    """Tạo ứng dụng FastAPI phục vụ mô hình Học Sâu (Deep Learning) trên Local hoặc Vercel."""
    app_settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 1. Khởi tạo cơ sở dữ liệu lịch sử (an toàn trên Vercel với /tmp)
        history_service = HistoryService(app_settings.history_db)
        try:
            history_service.initialize()
        except Exception as err:
            logger.warning("Không thể khởi tạo SQLite DB tại %s: %s", app_settings.history_db, err)
        app.state.settings = app_settings
        app.state.history_service = history_service

        # 2. Khởi tạo mô hình phân loại ảnh bằng Deep Learning (MobileNetV3 + MLP)
        app.state.classifier = None
        app.state.class_names = list(DEFAULT_CLASSES)
        try:
            if classifier_factory is not None:
                clf = classifier_factory(app_settings.dl_model_dir)
                clf.load()
                app.state.classifier = clf
                app.state.class_names = getattr(clf, "class_names", DEFAULT_CLASSES)
            else:
                from src_dl.predict_dl import FruitClassifierDL
                dl_path = app_settings.dl_model_dir / "classifier_dl.joblib"
                if dl_path.exists():
                    logger.info("Đang nạp mô hình Deep Learning từ %s", app_settings.dl_model_dir)
                    clf_dl = FruitClassifierDL(app_settings.dl_model_dir)
                    clf_dl.load()
                    app.state.classifier = clf_dl
                    app.state.class_names = clf_dl.class_names
                    logger.info("Đã nạp thành công mô hình Deep Learning (%d lớp)", len(clf_dl.class_names))
                else:
                    logger.warning("Không tìm thấy mô hình tại: %s", dl_path)
        except Exception as error:
            logger.warning("Mô hình phân loại ảnh chưa sẵn sàng: %s", error)
            app.state.model_error = str(error)

        yield

    application = FastAPI(
        title="Hệ Thống Phân Loại & Đánh Giá Chất Lượng Trái Cây bằng Học Sâu",
        summary="Ứng dụng thị giác máy tính nhận diện chủng loại, độ chín và đánh giá chất lượng trái cây bằng Deep Learning (MobileNetV3 + PyTorch CUDA).",
        description=(
            "Hệ thống tự động trích xuất đặc trưng sâu từ ảnh RGB (224x224), phân loại đồng thời 2 thuộc tính: "
            "Độ chín (Chín, Chưa chín, Hỏng) và Chất lượng thương phẩm (Đạt chuẩn Loại 1, Loại 2 hoặc Kém)."
        ),
        version="3.0.0",
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

    def _get_frontend_dir() -> Path:
        candidates = [
            app_settings.frontend_root,
            Path(__file__).resolve().parent / "frontend",
            Path.cwd() / "app" / "web" / "frontend",
            Path.cwd() / "frontend",
        ]
        for c in candidates:
            if c.is_dir() and (c / "index.html").is_file():
                return c
        return app_settings.frontend_root

    frontend = _get_frontend_dir()
    static_dir = frontend / "static"
    if static_dir.is_dir():
        application.mount(
            "/static", StaticFiles(directory=static_dir, check_dir=True), name="static"
        )
    if app_settings.reports_root.is_dir():
        application.mount(
            "/reports",
            StaticFiles(directory=app_settings.reports_root, check_dir=True),
            name="reports",
        )
    if app_settings.reports_dl_root.is_dir():
        application.mount(
            "/reports_dl",
            StaticFiles(directory=app_settings.reports_dl_root, check_dir=True),
            name="reports_dl",
        )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    @application.get("/api", response_class=HTMLResponse, include_in_schema=False)
    @application.get("/api/index", response_class=HTMLResponse, include_in_schema=False)
    @application.get("/api/index.py", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        f_dir = _get_frontend_dir()
        index_file = f_dir / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Không tìm thấy index.html")
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

    @application.get("/pages/{page_name}.html", response_class=HTMLResponse, include_in_schema=False)
    def page(page_name: str) -> HTMLResponse:
        if page_name not in PAGE_NAMES:
            raise HTTPException(status_code=404, detail="Không tìm thấy trang.")
        f_dir = _get_frontend_dir()
        page_file = f_dir / "pages" / f"{page_name}.html"
        if not page_file.exists():
            raise HTTPException(status_code=404, detail=f"Không tìm thấy {page_name}.html")
        return HTMLResponse(content=page_file.read_text(encoding="utf-8"))

    return application


app = create_app()
