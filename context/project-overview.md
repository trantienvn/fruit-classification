# Tổng quan dự án

- **Mục tiêu:** Phân loại ảnh trái cây bằng đặc trưng thị giác thủ công và
  machine learning truyền thống; cung cấp API/UI để dự đoán.
- **Môi trường:** Python 3.14; scikit-learn, Pillow/OpenCV/scikit-image,
  FastAPI và pytest.
- **Train mặc định:** Ảnh nguyên cảnh tại `data/processed/whole-scene/`,
  taxonomy Apple/Banana/Mango × Unripe/Ripe/Overripe, giữ split train/val/test.
  Dataset được xuất từ `data/raw/` bằng `python -m src.dataset`.
- **Chế độ khác:** `folder` đọc dataset Cam/Chuối/Xoài từ
  `data/processed/dataset/`; `yolo` crop bbox từ dataset YOLO.
- **Pipeline:** 51 feature, StandardScaler, 7 estimator scikit-learn, chọn
  trên validation và đánh giá test độc lập. Huấn luyện chỉ hỗ trợ CPU.
- **Ứng dụng:** `src/predict.py` cung cấp `FruitClassifier`; `app/web/` là
  FastAPI + HTML/CSS/JS. Streamlit vẫn còn như ứng dụng legacy.

Dataset và model/report artifacts là dữ liệu cục bộ; không thêm chúng vào Git.
