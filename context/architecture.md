# Kiến trúc hiện hành

Dự án sử dụng nhánh **Học Sâu (`src_dl/`)** làm trọng tâm chính, bỏ qua nhánh học máy truyền thống cũ (`src/`). Giao diện người dùng sử dụng thư mục **`frontend/`** ở thư mục gốc.

## 1. Nhánh Deep Learning chính (`src_dl/`)

- **Backbone trích xuất đặc trưng (`src_dl/backbone.py`)**: Sử dụng MobileNetV3-Small (pretrained ImageNet, đóng băng tham số - frozen) để trích xuất đặc trưng tự động.
- **Tiền xử lý & Tăng cường ảnh (`src_dl/preprocess_dl.py`)**: Resize ảnh về kích thước chuẩn 224×224, chuẩn hóa theo phân phối ImageNet (mean/std), áp dụng Data Augmentation (Random Horizontal Flip, Rotation, ColorJitter) trên tập train.
- **Mô hình phân loại (`src_dl/torch_classifier.py`)**: Tầng phân loại MLP / Scikit-learn Classifier trên vector embedding 576 chiều (hoặc phân loại PyTorch MLP có hỗ trợ GPU/CUDA).
- **Huấn luyện & Tái sử dụng (`src_dl/train_dl.py`)**: Huấn luyện phân loại 9 lớp (Cam, Chuối, Xoài × Chín, Chưa chín, Hỏng), hỗ trợ tiếp tục huấn luyện (`--resume`), theo dõi loss và xuất biểu đồ học tập (`loss_curve.png`).
- **Suy luận độc lập (`src_dl/predict_dl.py`)**: Đóng gói lớp `FruitClassifierDL` thực hiện suy luận từ ảnh tệp hoặc bytes đầu vào, tính toán xác suất phân phối 9 lớp.
- **Artifacts & Báo cáo**: Artifacts lưu tại `models_dl/`, biểu đồ và báo cáo thực nghiệm tại `reports_dl/`.

## 2. Giao diện Web & REST API

- **Web Frontend (`frontend/`)**: 
  - Nằm tại thư mục gốc `frontend/`, chứa giao diện người dùng Single Page Application (Dashboard, Nhận diện ảnh, Lịch sử, Thống kê, Thông tin mô hình).
  - Hỗ trợ suy luận trực tiếp trong trình duyệt qua **ONNX Runtime Web** (`frontend/models/`) và kéo thả / phím tắt `Ctrl + V` dán ảnh từ clipboard.
  - Cấu hình triển khai chuẩn với Vercel (`outputDirectory: "frontend"` trong `vercel.json`).
- **REST API Backend (`app/web/`)**:
  - Dựa trên FastAPI, nạp mô hình từ `models_dl/` qua `src_dl.predict_dl.FruitClassifierDL`.
  - Cung cấp các endpoint: `/api/predict`, `/api/health`, `/api/model-info`, `/api/history`, `/api/statistics`.
  - Phục vụ tĩnh trực tiếp thư mục `frontend/` ở thư mục gốc (thông qua `app/web/config.py`).

## 3. Nhánh cũ (`src/` - Legacy / Bỏ qua)
- Nhánh `src/` (trích xuất 51 đặc trưng thủ công color/texture/shape kết hợp scikit-learn) được giữ nguyên trạng làm tài liệu tham khảo lịch sử, không tham gia vào luồng xử lý chính hiện hành của dự án.
