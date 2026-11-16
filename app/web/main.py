"""FastAPI app, khởi tạo các model (Deep Learning ảnh & Machine Learning bảng) và phục vụ giao diện tĩnh.
Tương thích hoàn toàn với môi trường Local và Vercel Serverless Functions.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.dataset import CLASS_NAMES
from src.predict import FruitClassifier
from src_tabular.predict_tabular import FruitQualityClassifierTabular

try:
    from src_dl.predict_dl import FruitClassifierDL
except Exception as _dl_err:
    FruitClassifierDL = None

from app.web.config import Settings
from app.web.routes import (
    health,
    history,
    model_info,
    prediction,
    statistics,
    tabular,
)
from app.web.services.history_service import HistoryService

logger = logging.getLogger(__name__)
ClassifierFactory = Callable[[Path | str], Any]
PAGE_NAMES = {"dashboard", "recognition", "history", "statistics", "model-info"}


def create_app(
    settings: Settings | None = None,
    classifier_factory: ClassifierFactory | None = None,
) -> FastAPI:
    """Tạo app có thể cấu hình để kiểm thử và phục vụ cả 2 mô hình (DL & Tabular) trên Local hoặc Vercel."""
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

        # 2. Khởi tạo mô hình phân loại ảnh (Ưu tiên Deep Learning models_dl, fallback an toàn)
        app.state.classifier = None
        app.state.class_names = list(CLASS_NAMES)
        try:
            dl_path = app_settings.dl_model_dir / "classifier_dl.joblib"
            if FruitClassifierDL is not None and dl_path.exists():
                logger.info("Đang nạp mô hình Deep Learning từ %s", app_settings.dl_model_dir)
                clf_dl = FruitClassifierDL(app_settings.dl_model_dir)
                clf_dl.load()
                app.state.classifier = clf_dl
                app.state.class_names = clf_dl.class_names
                logger.info("Đã nạp thành công mô hình Deep Learning (%d lớp)", len(clf_dl.class_names))
            else:
                factory = classifier_factory or FruitClassifier
                logger.info("Nạp mô hình từ %s", app_settings.model_dir)
                classifier = factory(app_settings.model_dir)
                classifier.load()
                app.state.classifier = classifier
                app.state.class_names = classifier.get_info().get(
                    "class_names", list(CLASS_NAMES)
                )
        except Exception as error:
            logger.warning("Mô hình phân loại ảnh chưa sẵn sàng: %s", error)
            app.state.model_error = str(error)

        # 3. Khởi tạo mô hình học máy theo bảng số liệu (Tabular Quality ML)
        app.state.tabular_classifier = None
        try:
            tabular_path = app_settings.tabular_model_dir / "classifier_tabular.joblib"
            if tabular_path.exists():
                logger.info("Đang nạp mô hình học máy bảng từ %s", app_settings.tabular_model_dir)
                tab_clf = FruitQualityClassifierTabular(app_settings.tabular_model_dir)
                tab_clf.load()
                app.state.tabular_classifier = tab_clf
                logger.info("Đã nạp thành công mô hình học máy bảng: %s", tab_clf.config.get("best_model"))
            else:
                logger.warning("Không tìm thấy mô hình bảng tại %s", tabular_path)
        except Exception as error:
            logger.warning("Mô hình bảng chưa sẵn sàng: %s", error)
            app.state.tabular_model_error = str(error)

        yield

    application = FastAPI(
        title="Hệ Thống Phân Loại & Đánh Giá Chất Lượng Trái Cây",
        summary="Ứng dụng tích hợp 2 bản: Học sâu phân loại ảnh & Học máy đánh giá chất lượng theo bảng số liệu.",
        description=(
            "1. Bản Học Sâu (Deep Learning): Nhận ảnh JPG, PNG, BMP, WEBP -> Phân loại loại quả và độ chín / hỏng bằng MobileNetV3 + MLP.\n"
            "2. Bản Học Máy (Machine Learning): Nhận bảng 7 số liệu (Kích thước, Khối lượng, Độ ngọt, Độ giòn, Độ mọng, Độ axit, Độ chín) -> Đánh giá Tốt / Kém."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )
    application.state.settings = app_settings
    application.include_router(health.router, prefix="/api")
    application.include_router(prediction.router, prefix="/api")
    application.include_router(tabular.router, prefix="/api")
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
