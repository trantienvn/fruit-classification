# Thay đổi gần nhất

## 2026-11-17
- Đồng bộ định hướng phát triển theo yêu cầu người dùng:
  - Sử dụng toàn bộ quy trình Học Sâu trong `src_dl/` (MobileNetV3-Small trích xuất đặc trưng + MLP Head, lưu trữ tại `models_dl/`), bỏ qua mã nguồn ML truyền thống cũ trong `src/`.
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
