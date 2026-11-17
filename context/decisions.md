# Quyết định hiện hành

- **Công nghệ & pipeline chính**: Sử dụng Deep Learning với nhánh `src_dl/` (MobileNetV3-Small trích xuất đặc trưng + PyTorch MLP Head / Transfer Learning); bỏ qua pipeline Machine Learning truyền thống cũ trong `src/`.
- **Giao diện Web**: Sử dụng giao diện trong thư mục `frontend/` ở thư mục gốc (Single Page Application hỗ trợ ONNX Runtime Web in-browser và kết nối REST API FastAPI; tương thích Vercel outputDirectory `frontend`).
- Taxonomy: Cam / Chuối / Xoài (Orange / Banana / Mango) × Chín / Chưa chín / Hỏng (Ripe / Unripe / Spoiled).
- Artifact mô hình lưu tại `models_dl/`, báo cáo thực nghiệm tại `reports_dl/`.
- API/web dùng mô hình học sâu từ `models_dl/`; không lưu trữ ảnh upload vào cơ sở dữ liệu SQLite lịch sử để bảo đảm an toàn và hiệu năng.
