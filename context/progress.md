# Tiến độ

- **Hoàn thành:** pipeline scikit-learn CPU với taxonomy Apple/Banana/Mango ×
  Unripe/Ripe/Overripe, 51 đặc trưng, PCA 95%, CV stratified 5-fold,
  train-only augmentation, ablation, so sánh global/per-fruit và báo cáo.
- **Dữ liệu train:** whole-scene tại `data/processed/whole-scene/`; không có
  lựa chọn dataset mode trong CLI.
- **Kiểm thử gần nhất:** `python -m pytest tests -q`: 42 passed, 1 cảnh báo
  deprecation Starlette/httpx; CLI help và `git diff --check` đạt.
- Theo chỉ đạo người dùng, không chạy train lại sau bugfix cuối. Kết quả/artifact
  của lần train trước không được coi là kết quả của phiên bản code cuối.
- Web mặc định dùng `reports-whole-scene/`; có thể ghi đè bằng
  `FRUIT_REPORTS_DIR`. Trang **Thông tin mô hình**
  hiển thị các ảnh báo cáo hiện có.
- `src.audit` kiểm tra số lượng, ảnh lỗi, kích thước và duplicate chính xác;
  duplicate được lọc theo thứ tự test > val > train khi nạp dữ liệu.
