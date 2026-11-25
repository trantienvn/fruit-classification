# Tiến độ

- **Tiến độ:** Pipeline Học Sâu (src/) hoàn thiện. Pipeline ML truyền thống đã xóa.
- **Dữ liệu train:** whole-scene tại `data/processed/whole-scene/`; không có
  lựa chọn dataset mode trong CLI.
- **Kiểm thử gần nhất:** `python -m pytest tests -q`: 42 passed, 1 cảnh báo
  deprecation Starlette/httpx; CLI help và `git diff --check` đạt.
- Theo chỉ đạo người dùng, không chạy train lại sau bugfix cuối. Kết quả/artifact
  của lần train trước không được coi là kết quả của phiên bản code cuối.
- Web mặc định dùng `reports-whole-scene/`; có thể ghi đè bằng
  `FRUIT_REPORTS_DIR`. Trang **Thông tin mô hình**
  hiển thị các ảnh báo cáo hiện có.
- Mỗi lần train sinh thêm lưu đồ train/inference, phân bố lớp theo split, ví dụ
  augmentation thật, điểm CV/GridSearch, metrics theo lớp, thời gian và so sánh
  model toàn cục/theo quả; xem danh sách tại `docs/training.md`.
- Có lệnh tùy chọn `python -m src.train_epochs --epochs 50` để train SGD theo
  epoch; mỗi lần chạy lưu model, CSV lịch sử và biểu đồ Macro-F1/log loss trong
  thư mục run riêng. `--resume` tiếp tục checkpoint run mới nhất (hoặc chọn
  `RUN_ID`), cộng số epoch mới và cập nhật cùng CSV/biểu đồ; chỉ cho phép khi dữ
  liệu và cấu hình không đổi. Epoch run cũng xuất biểu đồ Loss riêng, confusion
  matrix, class metrics, ROC/AUC, data distribution và báo cáo test. Đã xác minh
  bằng dữ liệu tổng hợp, không chạy dataset thật.
- `src.audit` kiểm tra số lượng, ảnh lỗi, kích thước và duplicate chính xác;
  duplicate được lọc theo thứ tự test > val > train khi nạp dữ liệu.

## Nhánh Deep Learning (src/)

- **Thêm mới (TASK-DL01):** Tạo `src/` — pipeline CNN embedding + sklearn hoàn toàn độc lập.
- **Backbone:** MobileNetV3-Small pretrained (ImageNet), frozen, embedding 576 chiều.
- **Augmentation:** Thực hiện trên ảnh gốc (trước CNN), không phải trên embedding.
- **Tài liệu:** `readme_dl.md` (đã xóa sau migration; nội dung tích hợp vào README.md).
- **Phụ thuộc bổ sung:** `torch>=2.0`, `torchvision>=0.15`.
- **Artifact:** `models/`, `reports/`, `data/processed/dl_features/` (đã thêm `.gitignore`).
