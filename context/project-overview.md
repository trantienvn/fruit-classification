# Tổng quan dự án (Project Overview)

## 1. Thông tin chung
- **Tên đề tài**: Phân loại chất lượng và độ chín của trái cây.
- **Bối cảnh**: Đồ án học phần Học máy (Machine Learning) — Năm học 2026.
- **Giáo viên hướng dẫn**: TS. Ngô Hữu Huy.
- **Nhóm thực hiện (Nhóm 03)**:
  - Trần Văn Tiến (Trưởng nhóm / ML & Pipeline)
  - Đặng Xuân Thảo (Data Preprocessing & Evaluation)
  - Lý Văn Quyến (Feature Extraction & UI)

## 2. Mục tiêu dự án
Xây dựng một hệ thống hoàn chỉnh từ dữ liệu ảnh, tiền xử lý, trích xuất đặc trưng thị giác máy tính truyền thống, huấn luyện các thuật toán học máy phân loại độ chín và loại trái cây, đến đóng gói pipeline và giao diện web demo trực quan, phục vụ báo cáo và kiểm tra thực nghiệm.

## 3. Công nghệ sử dụng
- **Ngôn ngữ**: Python 3.14 (môi trường `.venv` Python 3.14.7 được quản lý qua `uv`).
- **Thư viện Machine Learning**: `scikit-learn` (v1.9+), `scipy`, `joblib`.
- **Thư viện xử lý ảnh**: `opencv-python-headless` (v4.14+), `scikit-image` (v0.26+), `Pillow` (v12.3+).
- **Thư viện tính toán & dữ liệu**: `numpy`, `pandas`.
- **Thư viện hiển thị & báo cáo**: `matplotlib`, `seaborn`.
- **Giao diện người dùng (Frontend & Backend tích hợp)**: `streamlit` (v1.66+).
- **Kiểm thử tự động**: `pytest` (v8.4+).

## 4. Phạm vi bài toán
Bài toán được xác định là **Phân loại 9 lớp kết hợp (Loại trái cây × Độ chín)**:
1. `Apple Unripe` — Táo chưa chín (Mã lớp: 0)
2. `Apple Ripe` — Táo chín (Mã lớp: 1)
3. `Apple Overripe` — Táo quá chín (Mã lớp: 2)
4. `Banana Unripe` — Chuối chưa chín (Mã lớp: 3)
5. `Banana Ripe` — Chuối chín (Mã lớp: 4)
6. `Banana Overripe` — Chuối quá chín (Mã lớp: 5)
7. `Mango Unripe` — Xoài chưa chín (Mã lớp: 6)
8. `Mango Ripe` — Xoài chín (Mã lớp: 7)
9. `Mango Overripe` — Xoài quá chín (Mã lớp: 8)

*Ghi chú*: Dataset thực tế không có nhãn thương mại riêng biệt (Class 1, Class 2, Extra Class) nên tập trung giải quyết bài toán độ chín theo 9 lớp kết hợp nêu trên.

## 5. Các thành phần chính của hệ thống
1. **Dataset Pipeline (`src/dataset.py`)**: Đọc dữ liệu ảnh và nhãn bounding box YOLO từ `data/raw/`, crop theo vùng trái cây và chuẩn hóa kích thước 128×128.
2. **Feature Extractor (`src/features.py`)**: Trích xuất 51 đặc trưng thủ công (Color Histogram RGB, Color Stats RGB & HSV, LBP Texture, GLCM Texture, Shape Features từ Otsu mask).
3. **Training & Model Selection (`src/train.py`)**: Huấn luyện 7 thuật toán học máy truyền thống (KNN, Naive Bayes, Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting), áp dụng GridSearch và đánh giá trên tập kiểm tra độc lập.
4. **Evaluation & Visualization (`src/evaluate.py`)**: Tính toán Accuracy, Macro/Weighted F1, Confusion Matrix, ROC-AUC Curves, biểu đồ so sánh mô hình, Feature Importance.
5. **Inference Engine (`src/predict.py`)**: Class `FruitClassifier` nạp model, scaler và config từ `models/`, tiền xử lý ảnh đầu vào và trả về dự đoán cùng độ tin cậy.
6. **Web Dashboard (`app/streamlit_app.py`)**: Giao diện Streamlit thiết kế theo `frontend-design`, hỗ trợ Upload & Predict, xem lịch sử dự đoán và kiểm tra thông tin kỹ thuật mô hình.

