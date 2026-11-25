# Thay đổi gần nhất

## 2026-11-18
- Tạo `README.md` chính thức mới tập trung toàn diện vào kiến trúc Học Sâu (`src/`), Edge Inference qua ONNX Runtime Web trên trình duyệt, các tính năng UI SPA hiện đại (`frontend/`).
- Cập nhật và chuẩn hóa toàn bộ tài liệu về số lượng ảnh của tập dữ liệu thực tế:
  - Khảo sát thực tế bằng `src.audit`: Tổng cộng chính xác **1.800 ảnh**, chia đều cho 9 lớp (200 ảnh/lớp).
  - Tỷ lệ phân hoạch chuẩn 70% / 15% / 15%: **1.260 ảnh train** (140 ảnh/lớp), **270 ảnh val** (30 ảnh/lớp), **270 ảnh test** (30 ảnh/lớp).
  - Khắc phục các con số cũ/lệch (1.801 ảnh, 271 val) xuất phát từ 1 ảnh dư trong `val/chuoi-chin` trước đây.
  - Đồng bộ số liệu trên `data/README.md`, `docs/training.md`, `docs/cnn-feature-extraction.md`, `docs/results.md`, `readme_dl.md`.

## 2026-11-17
- Đồng bộ định hướng phát triển theo yêu cầu người dùng:
  - Sử dụng toàn bộ quy trình Học Sâu trong `src/` (MobileNetV3-Small trích xuất đặc trưng + MLP Head, lưu trữ tại `models/`), bỏ qua mã nguồn ML truyền thống cũ.
  - Thiết lập giao diện Web mặc định trỏ về thư mục `frontend/` ở thư mục gốc (Single Page Application hỗ trợ ONNX Runtime Web in-browser và tương thích Vercel).
  - Cập nhật `app/web/config.py`, `context/architecture.md` và `context/decisions.md`.

## 2026-11-03
- Xác nhận Logistic Regression được huấn luyện cùng các thuật toán phân loại
  khác và xuất hiện trong biểu đồ `model_comparison.png`; thêm kiểm thử hồi quy.
- Web chọn thư mục biểu đồ theo `dataset_mode` của model; trang thông tin mô hình
  hiển thị ảnh so sánh thuật toán và các báo cáo đánh giá.
- Dataset train mặc định chuyển sang ảnh nguyên cảnh
  `data/processed/whole-scene/`; mode `folder`/`yolo` vẫn chọn được.
- Lệnh train chỉ còn pipeline scikit-learn CPU; bỏ tùy chọn CUDA/TorchMLP/resume
  và dependency PyTorch CUDA.
- Sửa GridSearch cho SVM calibration bằng clone estimator và tham số
  `estimator__*`.
- Rút gọn context, bỏ lịch sử task/session chi tiết và các mục mô tả tính năng
  không còn trong luồng được hỗ trợ.
