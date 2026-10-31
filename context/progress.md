# Tiến độ dự án (Project Progress)

> Cập nhật lần cuối: 2026-10-31

## 1. Trạng thái tổng quan

| Giai đoạn | Mô tả | Trạng thái |
|---|---|---|
| **Giai đoạn 1** | Khảo sát repository, thống nhất yêu cầu, thiết lập môi trường Python 3.14 + uv | **Hoàn thành 100%** |
| **Giai đoạn 2** | Hoàn thiện Dataset pipeline & Crop Bounding Box YOLO (2.507 mẫu) | **Hoàn thành 100%** |
| **Giai đoạn 3** | Trích xuất đặc trưng (Feature Extraction - 51 features) | **Hoàn thành 100%** |
| **Giai đoạn 4** | Huấn luyện mô hình (7 thuật toán + GridSearchCV + Lưu checkpoint) | **Hoàn thành 100%** (SVM Test Acc: 93.5%, F1: 93.71%) |
| **Giai đoạn 5** | Hoàn thiện Backend & Inference Engine (`FruitClassifier`) | **Hoàn thành 100%** |
| **Giai đoạn 6** | Hoàn thiện Frontend Web App (Streamlit UI đầy đủ 3 tabs) | **Hoàn thành 100%** |
| **Giai đoạn 7** | Tích hợp End-to-End & Testing (17/17 tests passed) | **Hoàn thành 100%** |
| **Giai đoạn 8** | Tài liệu hóa đầy đủ (`docs/`, `README.md`, `context/`) | **Hoàn thành 100%** |

---

## 2. Chi tiết các thành phần hệ thống

### Dữ liệu & Tiền xử lý (`src/dataset.py`)
- [x] Khảo sát và xử lý toàn bộ 2.227 ảnh và 2.507 mẫu gán nhãn theo 3 split:
  - Train: 1.559 ảnh (1.770 mẫu đối tượng sau crop)
  - Val: 445 ảnh (491 mẫu đối tượng sau crop)
  - Test: 223 ảnh (246 mẫu đối tượng sau crop)
- [x] Khắc phục lỗi đọc tên file chứa nhiều dấu chấm (`.with_suffix`) trong định dạng UUID của Roboflow.
- [x] Crop ảnh chính xác theo bounding box YOLO, chuẩn hóa về kích thước 128×128 pixel.

### Trích xuất đặc trưng (`src/features.py`)
- [x] Vector 51 đặc trưng vật lý kết hợp:
  - 24 histogram RGB chuẩn hóa.
  - 9 chỉ số thống kê màu (RGB Mean/Std, HSV Mean).
  - 10 đặc trưng kết cấu LBP (Local Binary Patterns).
  - 4 đặc trưng kết cấu ma trận đồng mức xám GLCM.
  - 4 đặc trưng hình dạng Otsu mask.
- [x] Đảm bảo tính nhất quán 100% giữa quá trình Training và Inference.

### Huấn luyện & Đánh giá (`src/train.py`, `src/evaluate.py`)
- [x] Huấn luyện và so sánh 7 thuật toán học máy:
  - **SVM (RBF):** Val F1 = 0.9494 $\rightarrow$ sau GridSearchCV ($C=50$): **Val F1 = 0.9546**
  - **Random Forest:** Val F1 = 0.9441 $\rightarrow$ sau GridSearchCV: **Val F1 = 0.9459**
  - **Gradient Boosting:** Val F1 = 0.9316 $\rightarrow$ sau GridSearchCV: **Val F1 = 0.9443**
  - **KNN:** Val F1 = 0.8976
  - **Logistic Regression:** Val F1 = 0.8455
  - **Decision Tree:** Val F1 = 0.8365
  - **Naive Bayes:** Val F1 = 0.6714
- [x] Đánh giá trên tập Test độc lập (246 mẫu):
  - **Test Accuracy:** **93.50%**
  - **Macro F1:** **0.9371**
  - **Macro AUC:** **0.9985**
- [x] Xuất đầy đủ biểu đồ thực nghiệm vào `reports/`:
  - `confusion_matrix.png`
  - `roc_curves.png`
  - `model_comparison.png`
  - `classification_report_SVM.txt`
- [x] Lưu trữ trọng số và cấu hình vào `models/`:
  - `classifier.joblib`, `scaler.joblib`, `config.json`.

### Suy luận & Ứng dụng Web (`src/predict.py`, `app/streamlit_app.py`)
- [x] Lớp `FruitClassifier` nạp mô hình đã huấn luyện, tiền xử lý và suy luận trong ~10 ms.
- [x] Giao diện Streamlit hoạt động hoàn chỉnh với 3 tabs:
  - **Dự đoán:** Tải ảnh, phân loại độ chín, thẻ độ tin cậy, phân bố xác suất 9 lớp.
  - **Lịch sử:** Lưu lại các lượt phân loại trong phiên kèm ảnh thu nhỏ.
  - **Thông tin model:** Hiển thị thông số kỹ thuật thực tế và biểu đồ thực nghiệm.

---

## 3. Trạng thái kiểm thử
- `pytest tests/ -v`: **17/17 passed (100%)**.
- `py_compile`: Tất cả module trong `src/`, `app/`, `tests/` biên dịch sạch sẽ không lỗi.
- Kiểm thử end-to-end máy chủ web Streamlit: Phản hồi mã HTTP `200`.

### Demo Streamlit (2026-10-31)
- Hoàn thành TASK-F01: thiết kế lại `app/streamlit_app.py` theo nhận diện “Vườn chín”, giữ nguyên ba chức năng dự đoán, lịch sử và thông tin mô hình.
- Kiểm tra trên trình duyệt bằng ảnh test thật: SVM trả về “Mango Overripe” với confidence 84,2%; lịch sử chỉ ghi một lượt dù Streamlit rerun.
- Trang thông tin hiển thị test accuracy 93,5%, Macro F1 0,9371 và các biểu đồ báo cáo.
- Kiểm tra ở viewport rộng 390 px: không xuất hiện tràn ngang.

### Khắc phục đường dẫn chạy ứng dụng (2026-10-31)
- Hoàn thành TASK-G01: Streamlit tự xác định gốc repo, nên import `src` và nạp model/báo cáo không phụ thuộc current working directory.
- Đã kiểm tra AppTest từ thư mục `app/` và toàn bộ 17 bài test.
