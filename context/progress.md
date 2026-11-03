# Tiến độ

- **Hoàn thành:** dataset loaders/export whole-scene, 51 features, 7 model
  pipeline CPU, evaluation/reporting, inference, FastAPI web và frontend.
- **Mặc định hiện tại:** whole-scene; `folder` và `yolo` vẫn là mode tùy chọn.
- **Thay đổi gần nhất:** bỏ CUDA/TorchMLP/resume khỏi lệnh train và gỡ PyTorch
  khỏi `requirements.txt`; context đã được tinh gọn ngày 2026-11-03.
- **Kiểm thử gần nhất:** Web API 11 passed; toàn bộ suite 33 passed, 1 failed
  do mặc định `folder` trong thay đổi chưa commit của `src/train.py` không khớp
  với test đang kỳ vọng `whole-scene`. `py_compile` và `git diff --check` đạt.
- Chưa chạy train lại trên dataset thật sau khi đổi mặc định. Dataset whole-scene
  cần được xuất trước nếu chưa có sẵn.
- Web tự chọn `reports/` hoặc `reports-whole-scene/` theo `dataset_mode` của
  model; có thể ghi đè bằng `FRUIT_REPORTS_DIR`. Trang **Thông tin mô hình**
  hiển thị các ảnh báo cáo hiện có.
