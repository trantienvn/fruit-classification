# Tiến độ

- **Hoàn thành:** dataset loaders/export whole-scene, 51 features, 7 model
  pipeline CPU, evaluation/reporting, inference, FastAPI web và frontend.
- **Mặc định hiện tại:** whole-scene; `folder` và `yolo` vẫn là mode tùy chọn.
- **Thay đổi gần nhất:** bỏ CUDA/TorchMLP/resume khỏi lệnh train và gỡ PyTorch
  khỏi `requirements.txt`; context đã được tinh gọn ngày 2026-11-03.
- **Kiểm thử gần nhất:** `python -m pytest tests -q` — 32 passed, 1 cảnh báo
  deprecation từ Starlette/httpx; py_compile, CLI help và `git diff --check`
  đạt.
- Chưa chạy train lại trên dataset thật sau khi đổi mặc định. Dataset whole-scene
  cần được xuất trước nếu chưa có sẵn.
