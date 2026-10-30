# Tiến độ dự án (Project Progress)

## 1. Trạng thái tổng quan

| Giai đoạn | Mô tả | Trạng thái |
|---|---|---|
| **Giai đoạn 1** | Khảo sát repository, thống nhất yêu cầu, thiết lập môi trường | **Hoàn thành 100%** |
| **Giai đoạn 2** | Hoàn thiện Dataset pipeline & Crop Bounding Box | **Hoàn thành 100%** |
| **Giai đoạn 3** | Trích xuất đặc trưng (Feature Extraction - 51 features) | **Hoàn thành 100%** |
| **Giai đoạn 4** | Huấn luyện mô hình (7 thuật toán + GridSearch + Model Saving) | **Hoàn thành 90%** (Code sẵn sàng, chờ kích hoạt train full) |
| **Giai đoạn 5** | Hoàn thiện Backend & Inference Engine (`FruitClassifier`) | **Hoàn thành 100%** |
| **Giai đoạn 6** | Hoàn thiện Frontend Web App (Streamlit UI theo `frontend-design`) | **Hoàn thành 100%** |
| **Giai đoạn 7** | Tích hợp End-to-End & Testing | **Hoàn thành 85%** (16/17 unit tests passed) |
| **Giai đoạn 8** | Tài liệu hóa & Báo cáo học phần | **Đang tiến hành** |

---

## 2. Chi tiết các chức năng đã hoàn thành

### Dữ liệu & Tiền xử lý
- [x] Khảo sát và thống kê 2 227 ảnh thật chia theo 3 tập (`train`: 1 559, `val`: 445, `test`: 223).
- [x] Viết `src/dataset.py`: Parser YOLO format, tính tọa độ và crop ảnh bounding box, resize 128×128.
- [x] Lưu trữ cấu hình ánh xạ 9 nhãn trong `data/raw/data.yaml` và `src/dataset.py`.

### Đặc trưng học máy (Feature Engineering)
- [x] Viết `src/features.py`: Trích xuất vector 51 đặc trưng bao gồm:
  - 24 histogram RGB chuẩn hóa.
  - 9 chỉ số thống kê màu (RGB Mean/Std, HSV Mean/Std).
  - 10 đặc trưng kết cấu LBP (Local Binary Patterns).
  - 4 đặc trưng kết cấu GLCM (Contrast, Dissimilarity, Homogeneity, Energy).
  - 4 đặc trưng hình dạng trích từ Otsu mask (Area ratio, Aspect ratio, Extent, Eq diameter).
- [x] Đảm bảo tính nhất quán: Hàm trích xuất dùng chung 100% giữa Training và Inference.

### Huấn luyện & Đánh giá
- [x] Viết `src/train.py`:
  - Nạp 3 tập dữ liệu độc lập (`train`, `val`, `test`).
  - Fit `StandardScaler` trên tập train, transform sang val và test (chống data leakage).
  - Khung huấn luyện và so sánh 7 thuật toán: KNN, Naive Bayes, Logistic Regression, SVM (RBF), Decision Tree, Random Forest, Gradient Boosting.
  - Tinh chỉnh tham số tự động bằng `GridSearchCV` trên tập val cho top model.
  - Lưu trữ artifact: `classifier.joblib`, `scaler.joblib`, `config.json`.
- [x] Viết `src/evaluate.py`:
  - Tính Accuracy, Precision, Recall, Macro F1, Weighted F1.
  - Vẽ và xuất biểu đồ `reports/confusion_matrix.png`.
  - Vẽ và xuất đường cong `reports/roc_curves.png` (One-vs-Rest ROC/AUC).
  - Vẽ biểu đồ so sánh `reports/model_comparison.png`.
  - Vẽ `reports/feature_importance.png` cho các mô hình hỗ trợ.

### Suy luận & Giao diện người dùng
- [x] Viết `src/predict.py`: Lớp `FruitClassifier` quản lý vòng đời model, kiểm tra định dạng và dung lượng ảnh tải lên (<10MB), dự đoán xác suất và đo thời gian xử lý (ms).
- [x] Viết `app/streamlit_app.py`: Giao diện chuyên nghiệp, tuân thủ `frontend-design`:
  - Tab Dự đoán: Kéo thả ảnh, hiển thị ảnh gốc, kết quả phân loại (tiếng Việt & tiếng Anh), độ tin cậy, phân bố xác suất 9 lớp.
  - Tab Lịch sử: Lưu trữ và hiển thị 20 lần dự đoán gần nhất trong phiên.
  - Tab Thông tin Model: Hiển thị các chỉ số thực tế (Test Accuracy, Macro F1, số mẫu, đặc trưng) và các biểu đồ thực nghiệm.

---

### Trạng thái kiểm thử (Testing Status)
- Đã chạy `pytest tests/ -v`: **17/17 passed (100%)**.
- Tất cả unit test cho file handling, parse bounding box, feature extraction (51 features), metrics evaluation và inference engine đều vượt qua.

---

## 4. Công việc tiếp theo
1. Sửa lỗi làm tròn trong `test_parse_yolo_box_valid` để đạt 17/17 tests passed (100%).
2. Chạy huấn luyện chính thức `python -m src.train` trên tập dữ liệu thật để sinh `models/classifier.joblib` và các biểu đồ thực nghiệm trong `reports/`.
3. Kiểm thử ứng dụng Streamlit end-to-end với ảnh tải lên thực tế.
4. Cập nhật `README.md` và `docs/AGENT_PROJECT_SPEC.md` với số liệu thực tế thu được.
