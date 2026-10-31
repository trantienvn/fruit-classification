# Hướng dẫn dự đoán và ứng dụng Web UI (Inference)

Tài liệu hướng dẫn sử dụng bộ máy suy luận (Inference Engine) và giao diện trực quan hóa kết quả dự đoán độ chín trái cây.

---

## 1. Sử dụng qua Python API (`FruitClassifier`)

Lớp `FruitClassifier` (trong `src/predict.py`) là lớp đóng gói chuẩn mực phục vụ suy luận độc lập, không phụ thuộc vào dữ liệu huấn luyện.

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

## 3. Khởi chạy Giao diện Web (Streamlit UI)

Ứng dụng web được xây dựng tại `app/streamlit_app.py`, thiết kế chuyên biệt theo phong cách trực quan cho hệ thống phân loại nông sản AI.

### Lệnh khởi chạy

```powershell
streamlit run app/streamlit_app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ mặc định `http://localhost:8501`.

### Các chức năng trên giao diện:

1. **Tab 1: Dự đoán (Prediction)**
   - Kéo và thả hoặc tải lên các tệp ảnh (`JPG`, `PNG`, `BMP`, `WEBP`, tối đa 10 MB).
   - Xem trước hình ảnh tải lên và thông tin kích thước.
   - Hiển thị kết quả: Loại trái cây (tiếng Việt), Mức độ chín, Thẻ độ tin cậy.
   - Biểu đồ thanh phần trăm xác suất phân bố trên toàn bộ 9 lớp.
   - Đo thời gian xử lý thực tế tính theo mili-giây.

2. **Tab 2: Lịch sử (History)**
   - Lưu trữ danh sách 20 lần dự đoán gần nhất trong phiên làm việc.
   - Hiển thị ảnh thu nhỏ, thời điểm dự đoán, nhãn lớp và độ tin cậy.
   - Nút xóa lịch sử linh hoạt.

3. **Tab 3: Thông tin mô hình (Model Info)**
   - Các thẻ KPI tổng quan: Test Accuracy (93.5%), Macro F1 (0.9371), Số mẫu huấn luyện (1.770) và Số đặc trưng (51).
   - Bảng thông tin thuật toán chi tiết và danh mục 9 lớp mục tiêu.
   - Trực quan hóa các biểu đồ thực nghiệm lấy trực tiếp từ thư mục `reports/` (Confusion Matrix, ROC Curves, So sánh mô hình).
