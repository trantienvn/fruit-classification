# Thay đổi gần nhất

## 2026-11-03
- Dataset train mặc định chuyển sang ảnh nguyên cảnh
  `data/processed/whole-scene/`; mode `folder`/`yolo` vẫn chọn được.
- Lệnh train chỉ còn pipeline scikit-learn CPU; bỏ tùy chọn CUDA/TorchMLP/resume
  và dependency PyTorch CUDA.
- Sửa GridSearch cho SVM calibration bằng clone estimator và tham số
  `estimator__*`.
- Rút gọn context, bỏ lịch sử task/session chi tiết và các mục mô tả tính năng
  không còn trong luồng được hỗ trợ.
