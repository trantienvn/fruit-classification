# Hướng dẫn dự đoán và ứng dụng Web UI (Inference)

Tài liệu hướng dẫn sử dụng bộ máy suy luận (Inference Engine) và giao diện trực quan hóa kết quả dự đoán độ chín trái cây.

---

## 1. Sử dụng qua Python API (`FruitClassifier`)

Lớp `FruitClassifier` (trong `src/predict.py`) là lớp đóng gói chuẩn mực phục vụ suy luận độc lập, không phụ thuộc vào dữ liệu huấn luyện.

Nếu model được huấn luyện bằng `--device cuda`, môi trường suy luận cần có PyTorch.
Model sẽ dùng CUDA nếu khả dụng và tự suy luận trên CPU nếu không có CUDA. Model
scikit-learn hiện có không cần GPU; cài đặt PyTorch trong requirements không làm
thay đổi backend suy luận của model đó.

### Khởi tạo và nạp mô hình

```python
from pathlib import Path
from PIL import Image
from src.predict import FruitClassifier

# Khởi tạo đối tượng trỏ đến thư mục chứa model
classifier = FruitClassifier(model_dir=Path("models"))

# Nạp model, scaler và cấu hình
classifier.load()

# Kiểm tra trạng thái sẵn sàng
print("Model ready:", classifier.is_ready)
print("Thông tin mô hình:", classifier.get_info()["model_name"])
```

### Dự đoán từ PIL Image

```python
image = Image.open("data/raw/images/test/-1_jpg.rf.d74fe43dc3420613d4d51e940c55d319.jpg")
result = classifier.predict(image)

print("Nhãn:", result["class_name"])             # Ví dụ: 'Mango Overripe'
print("Loại quả:", result["fruit"])              # 'Mango'
print("Độ chín:", result["ripeness"])            # 'Overripe'
print("Độ tin cậy:", f"{result['confidence']*100:.1f}%")
print("Thời gian xử lý:", f"{result['processing_time_ms']} ms")
```

Nếu train với `--dataset-mode whole-scene`, nạp đúng artifact ở
`models-whole-scene` để tiền xử lý ảnh upload nhất quán:

```python
classifier = FruitClassifier(model_dir=Path("models-whole-scene")).load()
```

### Dự đoán trực tiếp từ đường dẫn file hoặc chuỗi bytes

```python
# Từ đường dẫn file ảnh
result = classifier.predict_from_path("path/to/fruit.jpg")

# Từ chuỗi bytes (dùng cho API REST hoặc Streamlit file uploader)
with open("path/to/fruit.jpg", "rb") as f:
    result = classifier.predict_from_bytes(f.read())
```

### Sử dụng mô hình qua Singleton Pattern

Để tối ưu bộ nhớ và thời gian khởi động, hàm `get_classifier()` đảm bảo mô hình chỉ được nạp một lần duy nhất vào bộ nhớ:

```python
from src.predict import get_classifier

clf = get_classifier()
res = clf.predict_from_path("fruit.jpg")
```

---

## 2. Cấu trúc kết quả dự đoán (Inference Output)

Dict kết quả trả về từ `predict()` chứa đầy đủ các thông tin:

```json
{
  "class_id": 8,
  "class_name": "Mango Overripe",
  "fruit": "Mango",
  "ripeness": "Overripe",
  "confidence": 0.8418,
  "probabilities": {
    "Apple Unripe": 0.0012,
    "Apple Ripe": 0.0021,
    "Apple Overripe": 0.0035,
    "Banana Unripe": 0.0018,
    "Banana Ripe": 0.0152,
    "Banana Overripe": 0.0241,
    "Mango Unripe": 0.0084,
    "Mango Ripe": 0.1019,
    "Mango Overripe": 0.8418
  },
  "processing_time_ms": 10.7
}
```

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

`POST /api/predict` hỗ trợ JPG, JPEG, PNG, BMP và WEBP, tối đa 10 MB. API kiểm tra nội dung ảnh bằng Pillow; đổi tên file hoặc `Content-Type` không làm một file không phải ảnh trở thành hợp lệ. Response gồm `predicted_class`, `class_name`, `fruit_type`, `ripeness`, `processing_time_ms` và metadata lịch sử. Trường `confidence`/`probabilities` chỉ được gửi khi model thực sự có `predict_proba`; model hiện tại là SVM đã bật xác suất. API không ánh xạ kết quả thành hạng chất lượng thương mại vì dataset chỉ có 9 lớp loại quả × độ chín.

Ảnh upload chỉ được dùng cho inference và không lưu xuống đĩa. SQLite lưu tên file, nhãn, độ tin cậy nếu có, thời gian xử lý và timestamp UTC tại `app/web/data/history.sqlite3` (đường dẫn này bị Git bỏ qua). Thống kê dashboard lấy từ lịch sử cục bộ; metrics kiểm thử trên trang mô hình lấy từ `models/config.json` và có ý nghĩa khác với thống kê sử dụng.

Có thể đổi cấu hình mà không sửa code bằng các biến `FRUIT_MODEL_DIR`, `FRUIT_HISTORY_DB`, `FRUIT_FRONTEND_DIR`, `FRUIT_REPORTS_DIR` và `FRUIT_MAX_UPLOAD_BYTES`. Nếu model chưa tồn tại, API vẫn phục vụ giao diện và `/api/health` báo `degraded`; endpoint dự đoán trả HTTP 503.

Để web dùng model ảnh toàn cảnh, đặt `FRUIT_MODEL_DIR=models-whole-scene` và
`FRUIT_REPORTS_DIR=reports-whole-scene` trước khi khởi chạy. Chế độ này nhận
ảnh toàn cảnh và resize trực tiếp về 128×128; ảnh có nhiều loại quả cùng lúc
không nằm trong phạm vi nhãn mà model được huấn luyện.

### Các màn hình giao diện

- **Tổng quan:** trạng thái dịch vụ, metrics của model và lịch sử phân loại thực tế.
- **Nhận diện ảnh:** kéo-thả/chọn ảnh, xem trước, trạng thái xử lý, nhãn và xác suất nếu model hỗ trợ.
- **Lịch sử:** tìm, lọc nhãn, phân trang và xóa metadata đã lưu.
- **Thống kê:** phân bố lớp, nhóm trái cây, mức độ chín và hoạt động 7 ngày từ SQLite.
- **Thông tin mô hình:** cấu hình, 9 lớp, metrics test và các báo cáo có sẵn trong `reports/`.

## 4. Giao diện Streamlit cũ

`app/streamlit_app.py` được giữ lại để đối chiếu. Ứng dụng HTML/CSS/JavaScript không dùng Streamlit để render. Có thể chạy bản cũ bằng:

```powershell
streamlit run app/streamlit_app.py
```
