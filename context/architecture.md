# Kiến trúc hiện hành

1. `src/dataset.py` xuất ảnh nguyên cảnh từ các split YOLO sang
   `data/processed/whole-scene/{train,val,test}/`; ảnh đa class/nhãn lỗi bị
   loại và thống kê được ghi trong manifest. Train mặc định đọc thư mục này.
2. `src/features.py` trích xuất 51 đặc trưng màu, kết cấu và hình dạng.
3. `src/train.py` chuẩn hóa, huấn luyện 7 mô hình scikit-learn trên CPU, chọn
   theo validation, đánh giá test và lưu classifier/scaler/config.
4. Artifact mặc định: `models-whole-scene/`, báo cáo:
   `reports-whole-scene/`. Có thể chỉ định thư mục model và dữ liệu khác qua
   tham số CLI; pipeline train luôn đọc cấu trúc whole-scene.
5. `src/predict.py` nạp artifact; `app/web/` dùng classifier cho REST API và
   frontend. API contract nằm tại `contracts/api-contract.md`.

Không có GPU/TorchMLP trong luồng train được hỗ trợ. `src/torch_model.py` và
artifact Torch cũ, nếu có, không thuộc dependency hay quy trình train hiện tại.

## Nhánh Deep Learning song song (`src_dl/`)

- Sử dụng MobileNetV3-Small (pretrained ImageNet, đóng băng tham số - frozen) để trích xuất đặc trưng tự động.
- Đầu vào ảnh resize về 224x224, qua backbone sinh vector embedding 576 chiều (thay thế 51 đặc trưng thủ công của `src/features.py`).
- Quá trình augmentation (xoay, lật, chỉnh độ sáng) được thực hiện trên ảnh gốc trước khi trích xuất vector.
- Vector embedding được cache tại `data/processed/dl_features/`.
- Pipeline phân loại sử dụng các thuật toán scikit-learn (chuẩn hóa StandardScaler, PCA 95%, GridSearchCV) tương thích với logic so sánh thuật toán học máy.
- Artifact lưu tại `models_dl/` và `reports_dl/`, hoàn toàn độc lập và không ảnh hưởng tới `src/`.
