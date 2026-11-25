# Hướng dẫn dự đoán và ứng dụng Web UI (Inference)

Tài liệu hướng dẫn sử dụng bộ máy suy luận (Inference Engine) và giao diện trực quan hóa kết quả dự đoán độ chín trái cây.

---

## 1. Sử dụng qua Python API (`FruitClassifierDL`)

Lớp `FruitClassifierDL` (trong `src/predict.py`) là lớp đóng gói chuẩn mực phục vụ suy luận độc lập, không phụ thuộc vào dữ liệu huấn luyện.

Pipeline: CNN backbone (MobileNetV3-Small, frozen) → embedding 576 chiều → sklearn/MLP classifier. Artifact mặc định lưu tại `models/` và báo cáo tại `reports/`.

### Khởi tạo và nạp mô hình

```python
from pathlib import Path
from PIL import Image
from src.predict import FruitClassifierDL

classifier = FruitClassifierDL(model_dir=Path("models"))
classifier.load()

print("Model ready:", classifier.is_ready)
print("Thông tin mô hình:", classifier.get_info()["model_name"])
```

### Dự đoán từ PIL Image

```python
image = Image.open("anh-trai-cay.jpg")
result = classifier.predict(image)

print("Nhãn:", result["class_name"])
print("Loại quả:", result["fruit"])
print("Trạng thái:", result["ripeness"])
print("Độ tin cậy:", f"{result['confidence']*100:.1f}%")
print("Thời gian xử lý:", f"{result['processing_time_ms']} ms")
```

### Dự đoán trực tiếp từ đường dẫn file hoặc chuỗi bytes

```python
result = classifier.predict_from_path("path/to/fruit.jpg")

with open("path/to/fruit.jpg", "rb") as f:
    result = classifier.predict_from_bytes(f.read())
```

---

## 2. Cấu trúc kết quả dự đoán (Inference Output)

Dict kết quả trả về từ `predict()` chứa đầy đủ các thông tin:

```json
{
  "class_id": 8,
  "class_name": "Mango Spoiled",
  "fruit": "Mango",
  "ripeness": "Spoiled",
  "confidence": "<giá trị từ model>",
  "probabilities": "<các xác suất từ model>",
  "processing_time_ms": "<thời gian thực tế>"
}
```

Tên lớp trong đoạn JSON chỉ minh họa cấu trúc; xác suất và thời gian được lấy
trực tiếp từ model đang nạp.

---

## 3. Chạy Web Application HTML/CSS/JavaScript

Ứng dụng mới dùng FastAPI để phục vụ HTML/CSS/JavaScript và API cùng origin. Backend nạp `FruitClassifier` một lần khi khởi động; các request ảnh gọi trực tiếp `predict_from_bytes()`.

### Khởi chạy

Từ thư mục gốc repository:

```powershell
.\.venv\Scripts\python.exe -m app.web
```

Hoặc chạy Uvicorn trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.web.main:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000`. Tài liệu Swagger UI ở `/docs`; OpenAPI JSON được phục vụ tại `/openapi.json`. Snapshot contract trong `app/web/openapi.json` được tạo từ khai báo FastAPI bằng `python -m app.web.export_openapi` và được kiểm thử để không lệch với schema runtime.

### REST API

| Endpoint | Hành vi |
|---|---|
| `GET /api/health` | Trạng thái API, model và kho SQLite (`ready` hoặc `degraded`). |
| `POST /api/predict` | Nhận multipart field `file`, phân loại ảnh và ghi metadata vào lịch sử. |
| `GET /api/model-info` | Đọc tên model, nhãn, đặc trưng, PCA và metrics từ model/config hiện có. |
| `GET /api/history` | Đọc lịch sử; hỗ trợ `q`, `class_name`, `limit` (1–100) và `offset`. |
| `DELETE /api/history` | Xóa metadata lịch sử trên máy đang chạy backend. |
| `GET /api/statistics` | Thống kê thật từ các lượt đã ghi trong SQLite. |

`POST /api/predict` hỗ trợ JPG, JPEG, PNG, BMP và WEBP, tối đa 10 MB. API kiểm tra nội dung ảnh bằng Pillow; response gồm `predicted_class`, `class_name`, `fruit_type`, `ripeness`, `processing_time_ms`. Trường `confidence`/`probabilities` chỉ được gửi khi model thực sự có `predict_proba`. Taxonomy: Cam/Chuối/Xoài × Chín/Chưa chín/Hỏng (9 lớp, nhãn dạng `cam-chin`, `chuoi-hong`, v.v.).

Ảnh upload chỉ được dùng cho inference và không lưu xuống đĩa. SQLite lưu tên file, nhãn, độ tin cậy nếu có, thời gian xử lý và timestamp UTC tại `app/web/data/history.sqlite3` (đường dẫn này bị Git bỏ qua). Thống kê dashboard lấy từ lịch sử cục bộ; metrics kiểm thử trên trang mô hình lấy từ `models/config.json` và có ý nghĩa khác với thống kê sử dụng.

Có thể đổi cấu hình mà không sửa code bằng các biến `FRUIT_MODEL_DIR`, `FRUIT_HISTORY_DB`, `FRUIT_FRONTEND_DIR`, `FRUIT_REPORTS_DIR` và `FRUIT_MAX_UPLOAD_BYTES`. Nếu model chưa tồn tại, API vẫn phục vụ giao diện và `/api/health` báo `degraded`; endpoint dự đoán trả HTTP 503.

### Các màn hình giao diện

- **Tổng quan:** trạng thái dịch vụ, metrics của model và lịch sử phân loại thực tế.
- **Nhận diện ảnh:** kéo-thả/chọn ảnh, xem trước, trạng thái xử lý, nhãn và xác suất nếu model hỗ trợ.
- **Lịch sử:** tìm, lọc nhãn, phân trang và xóa metadata đã lưu.
- **Thống kê:** phân bố lớp, nhóm trái cây, trạng thái quả và hoạt động 7 ngày từ SQLite.
- **Thông tin mô hình:** cấu hình, 9 lớp, metrics test và các báo cáo có sẵn trong `reports/`.

## 4. Edge Inference qua ONNX Runtime Web

Ngoài REST API server, mô hình cũng được xuất sang ONNX để chạy trực tiếp trong trình duyệt (không cần server). Xem `export_onnx.py` và thư mục `frontend/models/`.
