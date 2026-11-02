"""Giao diện Streamlit — Phân loại trạng thái trái cây.

Thiết kế theo fruit-ml-project và frontend-design skill:
  - Bản sắc riêng: xanh lá cây tươi + trắng xanh + accent cam đất
  - Typography: Inter (sans), dữ liệu số nổi bật
  - Layout: sidebar nav + main content
  - Empty states rõ ràng, không số liệu giả
"""

from __future__ import annotations

import io
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import FruitClassifier, InvalidImageError, ModelNotReadyError

# ── Cấu hình trang ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Vườn chín · Trạng thái trái cây",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = PROJECT_ROOT / "models"

# ── CSS tùy chỉnh ────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
:root {
    --paper: #F2EFDF;
    --leaf: #31523D;
    --moss: #718354;
    --banana: #E5BE57;
    --ink: #25332A;
    --rule: #D4D4C2;
}

.stApp, [data-testid="stAppViewContainer"] {
    background: var(--paper);
    color: var(--ink);
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
}
[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1280px;
    padding: 2.5rem 3.5rem 4rem;
}
[data-testid="stSidebar"] {
    background: var(--leaf);
    border-right: 1px solid #49634E;
}
[data-testid="stSidebar"] * { color: var(--paper); }
[data-testid="stSidebar"] .stRadio label {
    font-size: 1rem;
    padding: 0.25rem 0;
}
[data-testid="stSidebar"] hr { border-color: #60725A; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.7rem;
    font-weight: 500;
    letter-spacing: -0.04em;
}

div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {
    color: var(--ink);
    font-family: Georgia, "Times New Roman", serif;
    font-weight: 500;
    letter-spacing: -0.035em;
}
div[data-testid="stMarkdownContainer"] h1 { line-height: 1.05; }
p, li, label, input, button { line-height: 1.55; }
*:focus-visible {
    outline: 3px solid var(--banana) !important;
    outline-offset: 3px !important;
}
input[type="radio"] { accent-color: var(--banana) !important; }

.orchard-hero {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 2rem;
    padding: 0.25rem 0 1.8rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--rule);
}
.orchard-hero h1 {
    max-width: 760px;
    margin: 0;
    font-size: clamp(2.6rem, 5vw, 4.7rem);
}
.orchard-hero p {
    max-width: 590px;
    margin: 0.9rem 0 0;
    color: #586458;
    font-size: 1.05rem;
}
.orchard-mark {
    min-width: 150px;
    padding: 0.8rem 0 0.2rem 1.2rem;
    border-left: 3px solid var(--banana);
    color: var(--leaf);
}
.orchard-mark strong {
    display: block;
    font: 500 1.65rem Georgia, "Times New Roman", serif;
}
.orchard-mark span { color: #586458; font-size: 0.82rem; }

.section-title {
    margin: 0 0 0.6rem;
    color: var(--leaf);
    font: 600 0.9rem "Segoe UI", sans-serif;
}
.upload-note {
    color: #586458;
    font-size: 0.9rem;
}
[data-testid="stFileUploaderDropzone"] {
    min-height: 142px;
    background: rgba(255,255,255,0.28);
    border: 1px dashed #82917C;
    border-radius: 3px;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--leaf);
    background: rgba(229,190,87,0.12);
}

.result-rule {
    padding: 0.2rem 0 1.1rem;
    border-bottom: 1px solid var(--rule);
}
.result-label {
    margin-bottom: 0.65rem;
    color: #586458;
    font-size: 0.85rem;
}
.result-name {
    margin: 0;
    color: var(--leaf);
    font: 500 clamp(2rem, 3.4vw, 3rem)/1.1 Georgia, "Times New Roman", serif;
    letter-spacing: -0.04em;
}
.result-english {
    margin-top: 0.45rem;
    color: #586458;
    font-size: 0.92rem;
}
.confidence-value {
    margin: 1rem 0 0;
    color: var(--ink);
    font: 500 1.1rem Georgia, "Times New Roman", serif;
}
.empty-result {
    min-height: 240px;
    padding: 1.5rem 0;
    border-top: 1px solid var(--rule);
    color: #586458;
}
.empty-result strong {
    display: block;
    margin-bottom: 0.35rem;
    color: var(--leaf);
    font: 500 1.65rem Georgia, "Times New Roman", serif;
}
.status-line {
    padding: 0.5rem 0;
    color: var(--paper);
    font-size: 0.84rem;
}
.status-dot { color: var(--banana); }

div[data-testid="stMetric"] {
    padding: 1rem 0;
    border-top: 1px solid var(--rule);
    border-bottom: 1px solid var(--rule);
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--leaf);
    font-family: Georgia, "Times New Roman", serif;
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color: #586458; }
.history-rule { border: 0; border-top: 1px solid var(--rule); margin: 0.75rem 0; }

button[kind="secondary"] {
    border-radius: 2px;
    border-color: var(--leaf);
    color: var(--leaf);
}
.stButton > button:focus-visible { box-shadow: 0 0 0 3px var(--banana); }
div[data-testid="stProgressBar"] > div > div { background: var(--moss); }

@media (max-width: 760px) {
    [data-testid="stAppViewContainer"] .main .block-container {
        padding: 1.5rem 1.1rem 3rem;
    }
    .orchard-hero { align-items: flex-start; flex-direction: column; gap: 1.1rem; }
    .orchard-hero h1 { font-size: clamp(2.5rem, 11vw, 3.7rem); }
    .orchard-mark { min-width: auto; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        scroll-behavior: auto !important;
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""

# ── Session state ─────────────────────────────────────────────────────────────

def _init_session() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "page" not in st.session_state:
        st.session_state.page = "Dự đoán"
    if "last_prediction_hash" not in st.session_state:
        st.session_state.last_prediction_hash = None


def _add_to_history(result: dict, image_bytes: bytes, image_hash: str) -> None:
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "class_name": result["class_name"],
        "fruit": result["fruit"],
        "ripeness": result["ripeness"],
        "confidence": result["confidence"],
        "processing_time_ms": result.get("processing_time_ms", 0),
        "image_bytes": image_bytes,
        "image_hash": image_hash,
    }
    if st.session_state.last_prediction_hash == image_hash:
        return
    st.session_state.history.insert(0, entry)
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[:20]
    st.session_state.last_prediction_hash = image_hash


# ── Model loading (cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_classifier() -> FruitClassifier | None:
    if not FruitClassifier.model_exists(MODEL_DIR):
        return None
    try:
        clf = FruitClassifier(MODEL_DIR)
        clf.load()
        return clf
    except Exception as e:
        st.error(f"Lỗi nạp model: {e}")
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## Vườn chín")
        st.markdown("Trạm kiểm tra trạng thái\ncam, chuối và xoài")
        st.markdown("---")

        pages = ["Dự đoán", "Lịch sử", "Mô hình"]
        selected_page = st.session_state.get("page", "Dự đoán")
        if selected_page == "Thông tin model":
            selected_page = "Mô hình"
        if selected_page not in pages:
            selected_page = "Dự đoán"
        page = st.radio("Chức năng", pages, index=pages.index(selected_page))

        st.markdown("---")
        clf = _load_classifier()
        if clf and clf.is_ready:
            info = clf.get_info()
            st.markdown(
                f'<div class="status-line"><span class="status-dot">●</span> '
                f'Mô hình {info.get("model_name", "N/A")} đang hoạt động</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-line"><span class="status-dot">●</span> '
                "Chưa tìm thấy mô hình</div>",
                unsafe_allow_html=True,
            )
            st.caption("Chạy `python -m src.train` để tạo mô hình.")

        st.markdown("---")
        st.caption("Nhóm 03 · Học máy 2026")
        st.caption("TS. Ngô Hữu Huy")

    return page


# ── Trang Dự đoán ─────────────────────────────────────────────────────────────

def _ripeness_label(ripeness: str) -> str:
    mapping = {
        "Unripe": "Chưa chín",
        "Ripe": "Chín",
        "Spoiled": "Hỏng",
        "Overripe": "Quá chín",
    }
    return mapping.get(ripeness, ripeness)


def _fruit_label(fruit: str) -> str:
    mapping = {"Orange": "Cam", "Banana": "Chuối", "Mango": "Xoài", "Apple": "Táo"}
    return mapping.get(fruit, fruit)


def _render_predict_page() -> None:
    st.markdown(
        """
        <section class="orchard-hero">
          <div>
            <h1>Kiểm tra trái cây<br>qua màu vỏ.</h1>
            <p>
              Chọn một ảnh trái cây để xem mô hình nhận diện loại quả và
              trạng thái. Kết quả được tạo trực tiếp từ ảnh bạn gửi.
            </p>
          </div>
          <div class="orchard-mark">
            <strong>03 loại quả</strong>
            <span>Cam · Chuối · Xoài</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    clf = _load_classifier()
    model_ready = clf is not None and clf.is_ready

    col_upload, col_result = st.columns([1.05, 0.95], gap="large")

    with col_upload:
        st.markdown('<p class="section-title">Ảnh cần kiểm tra</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Thả ảnh vào đây hoặc chọn từ thiết bị",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            max_upload_size=10,
        )
        st.markdown(
            '<p class="upload-note">JPG, PNG, BMP hoặc WEBP · tối đa 10 MB</p>',
            unsafe_allow_html=True,
        )

        image_bytes = None
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            if image_bytes is None:
                st.error("Không thể đọc nội dung ảnh đã tải lên.")
                return
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                st.image(img, width="stretch")
                st.caption(
                    f"{img.width} × {img.height} px  |  "
                    f"{len(image_bytes) / 1024:.1f} KB"
                )
            except (OSError, ValueError):
                st.error("Không thể đọc ảnh. Hãy kiểm tra định dạng file.")
                return

    with col_result:
        st.markdown('<p class="section-title">Kết quả kiểm định</p>', unsafe_allow_html=True)

        if uploaded is None:
            st.markdown(
                """
                <div class="empty-result">
                  <strong>Chờ ảnh đầu tiên</strong>
                  Kết quả phân loại sẽ xuất hiện tại đây sau khi bạn chọn ảnh.
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        if not model_ready:
            st.warning(
                "Chưa có mô hình để phân loại. Chạy "
                "`python -m src.train` để huấn luyện trước."
            )
            return

        image_hash = hashlib.sha256(image_bytes).hexdigest()
        cached_hash = st.session_state.get("displayed_prediction_hash")
        result = st.session_state.get("displayed_prediction")
        if cached_hash != image_hash or result is None:
            with st.spinner("Đang đọc màu sắc và kết cấu ảnh..."):
                try:
                    result = clf.predict_from_bytes(image_bytes)  # type: ignore[union-attr]
                except (InvalidImageError, ModelNotReadyError) as error:
                    st.error(str(error))
                    return
                except Exception as error:
                    st.error(f"Không thể phân loại ảnh: {error}")
                    return
            st.session_state.displayed_prediction_hash = image_hash
            st.session_state.displayed_prediction = result
            _add_to_history(result, image_bytes, image_hash)

        fruit_vi = _fruit_label(result["fruit"])
        ripeness_vi = _ripeness_label(result["ripeness"])
        confidence_pct = result["confidence"] * 100

        st.markdown(
            f"""
            <div class="result-rule">
              <div class="result-label">Mô hình nhận diện</div>
              <p class="result-name">{fruit_vi}<br>{ripeness_vi}</p>
              <div class="result-english">{result["class_name"]}</div>
              <p class="confidence-value">Độ tin cậy: {confidence_pct:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"Thời gian xử lý ảnh: {result.get('processing_time_ms', 0):.1f} ms"
        )

        if result.get("probabilities"):
            with st.expander("Xem xác suất của 9 nhãn", expanded=True):
                sorted_probs = sorted(
                    result["probabilities"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                for class_name, probability in sorted_probs:
                    st.markdown(f"**{class_name}** · {probability * 100:.1f}%")
                    st.progress(float(probability))
        else:
            st.progress(result["confidence"])

# ── Trang Lịch sử ─────────────────────────────────────────────────────────────

def _render_history_page() -> None:
    st.markdown(
        """
        <section class="orchard-hero">
          <div>
            <h1>Những lần kiểm tra</h1>
            <p>Các ảnh đã phân loại trong phiên làm việc này.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state.get("history", [])

    if not history:
        st.markdown(
            """
            <div class="empty-result">
              <strong>Chưa có ảnh nào được kiểm tra</strong>
              Chọn “Dự đoán” và tải ảnh lên để bắt đầu.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if st.button("Xóa toàn bộ lịch sử", type="secondary"):
        st.session_state.history = []
        st.session_state.last_prediction_hash = None
        st.session_state.displayed_prediction_hash = None
        st.session_state.displayed_prediction = None
        st.rerun()

    st.caption(f"{len(history)} ảnh gần nhất · tối đa 20 ảnh trong một phiên")

    for entry in history:
        col_img, col_info = st.columns([0.6, 3.4])
        with col_img:
            try:
                st.image(Image.open(io.BytesIO(entry["image_bytes"])), width=92)
            except (OSError, ValueError):
                st.caption("Không thể hiển thị ảnh")
        with col_info:
            fruit_vi = _fruit_label(entry["fruit"])
            ripeness_vi = _ripeness_label(entry["ripeness"])
            st.markdown(f"**{fruit_vi} · {ripeness_vi}**")
            st.caption(
                f"{entry['class_name']}  |  {entry['time']}  |  "
                f"Độ tin cậy {entry['confidence'] * 100:.1f}%"
            )
        st.markdown("<hr class='history-rule'>", unsafe_allow_html=True)


# ── Trang Thông tin model ─────────────────────────────────────────────────────

def _render_model_info_page() -> None:
    st.markdown(
        """
        <section class="orchard-hero">
          <div>
            <h1>Bên trong mô hình</h1>
            <p>Các thông số và kết quả đánh giá trên tập kiểm tra độc lập.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    clf = _load_classifier()
    if clf is None or not clf.is_ready:
        st.info("Chưa có mô hình. Chạy `python -m src.train` để huấn luyện.")
        return

    info = clf.get_info()

    test = info.get("test_metrics", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        acc = test.get("accuracy", None)
        st.metric("Độ chính xác test", f"{acc * 100:.1f}%" if acc is not None else "N/A")
    with col2:
        f1 = test.get("macro_f1", None)
        st.metric("Macro F1", f"{f1:.4f}" if f1 is not None else "N/A")
    with col3:
        st.metric("Ảnh đối tượng huấn luyện", info.get("n_train", "N/A"))
    with col4:
        st.metric("Đặc trưng", info.get("n_features", "N/A"))
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### Cấu hình")
        st.write(f"Thuật toán: **{info.get('model_name', 'N/A')}**")
        st.write(f"Số lớp: **{info.get('n_classes', 9)}**")
        st.write(f"PCA: **{'Có' if info.get('use_pca') else 'Không'}**")

    with col_right:
        st.markdown("### Chín nhãn")
        classes = info.get("class_names", [])
        st.write("  ·  ".join(classes))

    st.markdown("### Đánh giá mô hình")

    reports_dir = PROJECT_ROOT / "reports"
    plots = {
        "Ma trận nhầm lẫn": reports_dir / "confusion_matrix.png",
        "Đường cong ROC": reports_dir / "roc_curves.png",
        "So sánh mô hình": reports_dir / "model_comparison.png",
        "Mức độ quan trọng của đặc trưng": reports_dir / "feature_importance.png",
    }

    available = {name: path for name, path in plots.items() if path.is_file()}
    if not available:
        st.info("Chưa có biểu đồ. Chạy `python -m src.train` để tạo.")
    else:
        tab_names = list(available.keys())
        tabs = st.tabs(tab_names)
        for tab, (name, path) in zip(tabs, available.items()):
            with tab:
                st.image(str(path), width="stretch")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    page = _render_sidebar()
    st.session_state.page = page

    if page == "Dự đoán":
        _render_predict_page()
    elif page == "Lịch sử":
        _render_history_page()
    elif page == "Mô hình":
        _render_model_info_page()


if __name__ == "__main__":
    main()
