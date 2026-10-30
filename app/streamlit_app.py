"""Giao diện Streamlit — Phân loại chất lượng và độ chín trái cây.

Thiết kế theo fruit-ml-project và frontend-design skill:
  - Bản sắc riêng: xanh lá cây tươi + trắng xanh + accent cam đất
  - Typography: Inter (sans), dữ liệu số nổi bật
  - Layout: sidebar nav + main content
  - Empty states rõ ràng, không số liệu giả
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from src.predict import FruitClassifier, InvalidImageError, ModelNotReadyError

# ── Cấu hình trang ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FruitScan AI",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = Path("models")

# ── CSS tùy chỉnh ────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* Palette:
   --leaf:     #2D6A4F  (xanh lá sẫm)
   --fresh:    #52B788  (xanh lá tươi)
   --mist:     #D8F3DC  (xanh nhạt)
   --soil:     #B7572A  (cam đất — accent)
   --sand:     #FAF7F2  (nền trắng kem)
   --ink:      #1B2E25  (chữ tối)
   --muted:    #6B8F71  (chữ phụ)
*/

/* Reset Streamlit defaults */
[data-testid="stAppViewContainer"] {
    background-color: #FAF7F2;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #1B2E25;
}
[data-testid="stSidebar"] * {
    color: #D8F3DC !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #D8F3DC !important;
    font-size: 0.95rem;
    padding: 4px 0;
}
[data-testid="stSidebar"] hr {
    border-color: #2D6A4F;
}

/* Header */
.fs-header {
    background: linear-gradient(135deg, #2D6A4F 0%, #52B788 100%);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.fs-header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.fs-header p {
    font-size: 0.9rem;
    margin: 0.3rem 0 0;
    opacity: 0.85;
}

/* Metric cards */
.fs-metric {
    background: white;
    border: 1px solid #D8F3DC;
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    text-align: center;
}
.fs-metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #2D6A4F;
    line-height: 1.1;
}
.fs-metric-label {
    font-size: 0.78rem;
    color: #6B8F71;
    margin-top: 0.2rem;
    text-transform: none;
}

/* Result card */
.fs-result {
    background: white;
    border-left: 4px solid #52B788;
    border-radius: 0 10px 10px 0;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
}
.fs-result-main {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1B2E25;
}
.fs-result-sub {
    font-size: 0.9rem;
    color: #6B8F71;
    margin-top: 0.3rem;
}
.fs-confidence {
    display: inline-block;
    background: #D8F3DC;
    color: #2D6A4F;
    font-weight: 600;
    font-size: 0.85rem;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    margin-top: 0.5rem;
}

/* Upload zone */
.fs-upload-hint {
    font-size: 0.82rem;
    color: #6B8F71;
    margin-top: 0.5rem;
}

/* Empty state */
.fs-empty {
    text-align: center;
    padding: 3rem 1rem;
    color: #6B8F71;
}
.fs-empty-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.fs-empty-title { font-size: 1.1rem; font-weight: 600; color: #2D6A4F; }
.fs-empty-body { font-size: 0.85rem; margin-top: 0.4rem; }

/* Model status badges */
.fs-badge-ok {
    background: #D8F3DC; color: #2D6A4F;
    border-radius: 20px; padding: 0.15rem 0.6rem;
    font-size: 0.78rem; font-weight: 600;
}
.fs-badge-warn {
    background: #FDE8D8; color: #B7572A;
    border-radius: 20px; padding: 0.15rem 0.6rem;
    font-size: 0.78rem; font-weight: 600;
}

/* Probability bar */
.fs-prob-row { margin: 0.35rem 0; }
.fs-prob-label { font-size: 0.78rem; color: #1B2E25; margin-bottom: 2px; }
.fs-prob-bar-bg {
    background: #F0F0F0; border-radius: 4px; height: 8px; width: 100%;
}
.fs-prob-bar-fill {
    background: #52B788; border-radius: 4px; height: 8px;
}

/* History table */
.fs-hist-row {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.6rem 0; border-bottom: 1px solid #F0EDE8;
    font-size: 0.85rem;
}

/* Divider */
.fs-divider { border: none; border-top: 1px solid #E8EDE8; margin: 1.2rem 0; }
</style>
"""

# ── Session state ─────────────────────────────────────────────────────────────

def _init_session() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts
    if "page" not in st.session_state:
        st.session_state.page = "Dự đoán"


def _add_to_history(result: dict, image_bytes: bytes) -> None:
    import datetime
    entry = {
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "class_name": result["class_name"],
        "fruit": result["fruit"],
        "ripeness": result["ripeness"],
        "confidence": result["confidence"],
        "processing_time_ms": result.get("processing_time_ms", 0),
        "image_bytes": image_bytes,
    }
    st.session_state.history.insert(0, entry)
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[:20]


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
        st.markdown("## 🍃 FruitScan AI")
        st.markdown("Phân loại độ chín trái cây\nbằng học máy truyền thống")
        st.markdown("---")

        page = st.radio(
            "Chức năng",
            ["Dự đoán", "Lịch sử", "Thông tin model"],
            index=["Dự đoán", "Lịch sử", "Thông tin model"].index(
                st.session_state.get("page", "Dự đoán")
            ),
        )

        st.markdown("---")
        clf = _load_classifier()
        if clf and clf.is_ready:
            st.markdown('<span class="fs-badge-ok">● Model sẵn sàng</span>', unsafe_allow_html=True)
            info = clf.get_info()
            st.caption(f"Thuật toán: {info.get('model_name', 'N/A')}")
        else:
            st.markdown('<span class="fs-badge-warn">● Chưa có model</span>', unsafe_allow_html=True)
            st.caption("Chạy `python -m src.train` để huấn luyện.")

        st.markdown("---")
        st.caption("Nhóm 03 · Học máy 2026")
        st.caption("TS. Ngô Hữu Huy")

    return page


# ── Trang Dự đoán ─────────────────────────────────────────────────────────────

def _ripeness_label(ripeness: str) -> str:
    mapping = {"Unripe": "Chưa chín", "Ripe": "Chín", "Overripe": "Quá chín"}
    return mapping.get(ripeness, ripeness)


def _fruit_label(fruit: str) -> str:
    mapping = {"Apple": "Táo", "Banana": "Chuối", "Mango": "Xoài"}
    return mapping.get(fruit, fruit)


def _render_predict_page() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="fs-header">
      <h1>🔍 Phân loại độ chín trái cây</h1>
      <p>Tải lên ảnh táo, chuối hoặc xoài — hệ thống sẽ phân loại độ chín tự động.</p>
    </div>
    """, unsafe_allow_html=True)

    clf = _load_classifier()
    model_ready = clf is not None and clf.is_ready

    col_upload, col_result = st.columns([1, 1], gap="large")

    # ── Cột upload ──
    with col_upload:
        st.subheader("Chọn ảnh")
        uploaded = st.file_uploader(
            "Kéo thả hoặc nhấn để chọn",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )
        st.markdown(
            '<p class="fs-upload-hint">Định dạng: JPG, PNG, BMP, WEBP · Tối đa 10 MB<br>'
            'Loại trái cây hỗ trợ: Táo, Chuối, Xoài</p>',
            unsafe_allow_html=True,
        )

        if uploaded is not None:
            image_bytes = uploaded.read()
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                st.image(img, caption="Ảnh đã tải lên", use_container_width=True)
                st.caption(f"Kích thước: {img.width}×{img.height}px · {len(image_bytes)/1024:.1f} KB")
            except Exception:
                st.error("Không thể đọc ảnh. Hãy kiểm tra định dạng file.")
                return

    # ── Cột kết quả ──
    with col_result:
        st.subheader("Kết quả dự đoán")

        if uploaded is None:
            st.markdown("""
            <div class="fs-empty">
              <div class="fs-empty-icon">🍎</div>
              <div class="fs-empty-title">Chưa có ảnh</div>
              <div class="fs-empty-body">Tải ảnh lên bên trái để bắt đầu phân loại.</div>
            </div>
            """, unsafe_allow_html=True)
            return

        if not model_ready:
            st.warning(
                "⚠️ Model chưa được huấn luyện.\n\n"
                "Chạy lệnh sau để huấn luyện:\n```\npython -m src.train\n```"
            )
            return

        # Chạy dự đoán
        with st.spinner("Đang phân tích ảnh..."):
            try:
                result = clf.predict_from_bytes(image_bytes)  # type: ignore[union-attr]
            except InvalidImageError as e:
                st.error(f"Ảnh không hợp lệ: {e}")
                return
            except ModelNotReadyError as e:
                st.error(f"Model chưa sẵn sàng: {e}")
                return
            except Exception as e:
                st.error(f"Lỗi không xác định: {e}")
                return

        # Hiển thị kết quả chính
        fruit_vi = _fruit_label(result["fruit"])
        ripeness_vi = _ripeness_label(result["ripeness"])
        confidence_pct = result["confidence"] * 100

        st.markdown(f"""
        <div class="fs-result">
          <div class="fs-result-main">🍃 {fruit_vi} — {ripeness_vi}</div>
          <div class="fs-result-sub">{result['class_name']}</div>
          <div class="fs-confidence">Độ tin cậy: {confidence_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"⏱ Thời gian xử lý: {result.get('processing_time_ms', 0):.1f} ms")

        # Xác suất từng lớp
        if result.get("probabilities"):
            st.markdown("**Xác suất theo lớp**")
            sorted_probs = sorted(
                result["probabilities"].items(), key=lambda x: x[1], reverse=True
            )
            for cls_name, prob in sorted_probs:
                pct = prob * 100
                bar_width = int(pct)
                # Highlight class được chọn
                style = "background: #2D6A4F;" if cls_name == result["class_name"] else ""
                st.markdown(f"""
                <div class="fs-prob-row">
                  <div class="fs-prob-label">{cls_name}</div>
                  <div class="fs-prob-bar-bg">
                    <div class="fs-prob-bar-fill" style="width:{bar_width}%;{style}"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"{pct:.1f}%")
        else:
            # Không có probabilities — dùng progress bar Streamlit
            st.progress(result["confidence"])

        # Thêm vào lịch sử
        _add_to_history(result, image_bytes)


# ── Trang Lịch sử ─────────────────────────────────────────────────────────────

def _render_history_page() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.subheader("📋 Lịch sử dự đoán")

    history = st.session_state.get("history", [])

    if not history:
        st.markdown("""
        <div class="fs-empty">
          <div class="fs-empty-icon">📂</div>
          <div class="fs-empty-title">Chưa có lịch sử</div>
          <div class="fs-empty-body">Hãy thực hiện dự đoán ở tab "Dự đoán" trước.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    if st.button("Xóa toàn bộ lịch sử", type="secondary"):
        st.session_state.history = []
        st.rerun()

    st.caption(f"Hiển thị {len(history)} lần dự đoán gần nhất (tối đa 20)")
    st.divider()

    for i, entry in enumerate(history):
        col_img, col_info = st.columns([1, 3])
        with col_img:
            try:
                st.image(Image.open(io.BytesIO(entry["image_bytes"])), width=80)
            except Exception:
                st.write("🖼")
        with col_info:
            fruit_vi = _fruit_label(entry["fruit"])
            ripeness_vi = _ripeness_label(entry["ripeness"])
            st.markdown(
                f"**{fruit_vi} — {ripeness_vi}** &nbsp; "
                f"<span style='color:#6B8F71'>{entry['class_name']}</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"🕐 {entry['time']} · "
                f"Tin cậy: {entry['confidence']*100:.1f}% · "
                f"⏱ {entry['processing_time_ms']:.1f} ms"
            )
        st.divider()


# ── Trang Thông tin model ─────────────────────────────────────────────────────

def _render_model_info_page() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.subheader("⚙️ Thông tin mô hình")

    clf = _load_classifier()
    if clf is None or not clf.is_ready:
        st.markdown("""
        <div class="fs-empty">
          <div class="fs-empty-icon">🤖</div>
          <div class="fs-empty-title">Chưa có model</div>
          <div class="fs-empty-body">
            Chạy lệnh sau để huấn luyện:<br>
            <code>python -m src.train</code>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    info = clf.get_info()

    # Metrics nổi bật
    test = info.get("test_metrics", {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        acc = test.get("accuracy", None)
        val = f"{acc*100:.1f}%" if acc is not None else "N/A"
        st.markdown(f'<div class="fs-metric"><div class="fs-metric-value">{val}</div><div class="fs-metric-label">Test Accuracy</div></div>', unsafe_allow_html=True)
    with col2:
        f1 = test.get("macro_f1", None)
        val = f"{f1:.4f}" if f1 is not None else "N/A"
        st.markdown(f'<div class="fs-metric"><div class="fs-metric-value">{val}</div><div class="fs-metric-label">Macro F1</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="fs-metric"><div class="fs-metric-value">{info.get("n_train", "N/A")}</div><div class="fs-metric-label">Mẫu huấn luyện</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="fs-metric"><div class="fs-metric-value">{info.get("n_features", "N/A")}</div><div class="fs-metric-label">Đặc trưng</div></div>', unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Thông tin kỹ thuật**")
        st.write(f"- **Thuật toán:** {info.get('model_name', 'N/A')}")
        st.write(f"- **Số lớp:** {info.get('n_classes', 9)}")
        st.write(f"- **Dùng PCA:** {'Có' if info.get('use_pca') else 'Không'}")
        st.write(f"- **Số features:** {info.get('n_features', 'N/A')}")

    with col_right:
        st.markdown("**Nhãn dự đoán**")
        classes = info.get("class_names", [])
        for i, name in enumerate(classes):
            st.write(f"- **{i}**: {name}")

    # Biểu đồ nếu có
    st.divider()
    st.markdown("**Biểu đồ kết quả**")

    reports_dir = Path("reports")
    plots = {
        "Confusion Matrix": reports_dir / "confusion_matrix.png",
        "ROC Curves": reports_dir / "roc_curves.png",
        "So sánh mô hình": reports_dir / "model_comparison.png",
        "Feature Importance": reports_dir / "feature_importance.png",
    }

    available = {name: path for name, path in plots.items() if path.is_file()}
    if not available:
        st.info("Chưa có biểu đồ. Chạy `python -m src.train` để tạo.")
    else:
        tab_names = list(available.keys())
        tabs = st.tabs(tab_names)
        for tab, (name, path) in zip(tabs, available.items()):
            with tab:
                st.image(str(path), use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session()
    page = _render_sidebar()
    st.session_state.page = page

    if page == "Dự đoán":
        _render_predict_page()
    elif page == "Lịch sử":
        _render_history_page()
    elif page == "Thông tin model":
        _render_model_info_page()


if __name__ == "__main__":
    main()
