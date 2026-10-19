"""Demo giao diện; không giả lập kết quả khi chưa có mô hình."""

from pathlib import Path

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Phân loại trái cây", page_icon="🍎")
st.title("Phân loại chất lượng và độ chín của trái cây")
st.write("Tải ảnh lên để chuẩn bị cho bước dự đoán. Cần checkpoint đã huấn luyện để có kết quả thật.")

uploaded = st.file_uploader("Chọn ảnh trái cây", type=["jpg", "jpeg", "png", "bmp", "webp"])
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Ảnh đã tải lên", use_container_width=True)

checkpoint = Path("models/fruit_classifier.pth")
if checkpoint.is_file():
    st.info("Đã tìm thấy checkpoint, nhưng bộ khung hiện chưa triển khai bộ nạp mô hình.")
else:
    st.warning("Chưa có checkpoint. Hãy chuẩn bị dataset theo data/README.md; ứng dụng chưa thể dự đoán.")

