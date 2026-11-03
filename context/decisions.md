# Quyết định hiện hành

- Python 3.14; xử lý ảnh và ML truyền thống với scikit-learn.
- Mặc định dùng whole-scene từ `data/processed/whole-scene/`, taxonomy
  Apple/Banana/Mango × Unripe/Ripe/Overripe. Tạo dữ liệu bằng:
  `python -m src.dataset --source-dir data/raw --output-dir data/processed/whole-scene`.
- `folder` là lựa chọn tường minh cho dataset Cam/Chuối/Xoài ×
  Unripe/Ripe/Spoiled; `yolo` là lựa chọn tường minh để crop bbox.
- Train chỉ chạy CPU; không hỗ trợ CUDA, TorchMLP hoặc resume.
- Giữ nguyên split; không dùng test để chọn model/tuning. Không commit dataset
  local hoặc model/report artifacts.
- Artifact và báo cáo whole-scene tách riêng để không ghi đè model folder.
- API/web dùng model artifact được cấu hình; không lưu ảnh upload vào lịch sử.
