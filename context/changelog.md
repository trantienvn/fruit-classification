# Nhật ký thay đổi (Changelog)

Toàn bộ các thay đổi mã nguồn và tài liệu của dự án được ghi nhận theo chuẩn [Keep a Changelog](https://keepachangelog.com/).

## [2026-10-30]

### Đã thêm (Added)
- Thư mục `context/` tuân thủ skill `context-manager`: `README.md`, `project-overview.md`, `architecture.md`, `decisions.md`, `progress.md`, `issues.md`, `changelog.md`, `sessions/2026-10-30.md`.
- `src/dataset.py`: Đọc dataset YOLO, phân tích bounding box, crop đối tượng và resize 128×128.
- `src/features.py`: Trích xuất 51 đặc trưng (Color Histogram, Color Statistics RGB/HSV, LBP Texture, GLCM Texture, Shape metrics từ Otsu mask).
- `src/train.py`: Pipeline huấn luyện 7 mô hình học máy truyền thống, kiểm thử độc lập, GridSearchCV, lưu mô hình joblib và metadata `config.json`.
- `src/evaluate.py`: Bộ công cụ tính metrics và xuất biểu đồ báo cáo (Confusion Matrix, ROC curves, Model Comparison, Feature Importance).
- `src/predict.py`: Bộ nạp mô hình và suy luận thời gian thực `FruitClassifier` phục vụ dự đoán ảnh độc lập hoặc qua Web UI.
- `app/streamlit_app.py`: Giao diện Web phân loại trái cây hiện đại, tùy biến CSS theo skill `frontend-design` (bản sắc nông nghiệp công nghệ cao, màu xanh lá đậm & tươi, nền kem, đầy đủ chức năng dự đoán, lịch sử và thông tin mô hình).
- `tests/test_pipeline.py`: Bộ kiểm thử tự động toàn diện bao gồm dataset parsing, tính toán đặc trưng, đánh giá mô hình và engine dự đoán.

### Đã thay đổi (Changed)
- `requirements.txt`: Thay thế PyTorch/torchvision bằng các thư viện học máy truyền thống và xử lý ảnh (`scikit-learn`, `scikit-image`, `opencv-python-headless`, `scipy`, `seaborn`, `joblib`, `streamlit`).
- Cấu hình môi trường: Tạo virtual environment `.venv` chuẩn Python 3.14.7 thông qua công cụ `uv`.

### Đã sửa (Fixed)
- Phát hiện lỗi làm tròn số thực trong việc chuyển đổi tọa độ YOLO pixel (`int()` gây mất số liệu, sửa bằng `int(round(...))`).
- Xử lý vấn đề môi trường sandbox chặn truy cập Python 3.14 của hệ thống bằng việc tạo venv nội bộ trong thư mục dự án.

