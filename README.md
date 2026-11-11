# Phân loại trạng thái trái cây

Dự án phân loại ảnh trái cây thành 9 lớp. Pipeline mặc định dùng dataset
Cam/Chuối/Xoài × Unripe/Ripe/Spoiled tại `data/processed/dataset/`. Pipeline
tách vùng quả bằng OpenCV, trích xuất đặc trưng màu/kết cấu/hình dạng và chỉ
huấn luyện các mô hình machine learning truyền thống bằng scikit-learn trên CPU.

## Dataset

Kiểm tra số lượng lớp, ảnh lỗi, kích thước và ảnh trùng bằng:

```powershell
python -m src.audit --data-dir data/processed/dataset
```

Dataset classification tại `data/processed/dataset/` giữ nguyên các split đã
cung cấp. Cấu trúc lớp, số lượng ảnh và nguồn dữ liệu được mô tả trong
[data/README.md](data/README.md). Ảnh trùng chính xác được loại khỏi train nếu
đã có ở validation/test; test được ưu tiên giữ nguyên.

Dataset YOLO **Fruit Ripeness** do Huzaifa công bố trên
[Roboflow Universe](https://universe.roboflow.com/huzaifa-qgpco/fruit-ripeness-unjex-xsudt),
sử dụng theo giấy phép
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Giấy phép cho phép sử dụng, chia sẻ và chỉnh sửa, kể cả cho mục đích thương
mại, với điều kiện ghi công phù hợp, dẫn nguồn, liên kết giấy phép và nêu rõ
các thay đổi.

Dataset YOLO cũ được lọc về 9 lớp Apple, Banana, Mango; các lớp `Rotten` và
loại quả khác không nằm trong ánh xạ cũ. Khi sử dụng hoặc phân phối lại bộ cũ,
hãy ghi công tác giả Huzaifa, dẫn
[nguồn dataset](https://universe.roboflow.com/huzaifa-qgpco/fruit-ripeness-unjex-xsudt),
liên kết [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) và nêu rõ
các thay đổi.

## Yêu cầu

- Windows 10/11, Python 3.14 và [`uv`](https://github.com/astral-sh/uv).
- Pipeline dùng scikit-learn trên CPU, không yêu cầu GPU, PyTorch hoặc trọng số
  pretrained.
- Có kết nối mạng để cài các thư viện trong `requirements.txt`.

## Cài đặt trên Windows

Cài `uv` nếu chưa có (một lần duy nhất):

```powershell
winget install astral-sh.uv
```

Mở PowerShell tại thư mục dự án, tạo môi trường ảo Python 3.14 và cài thư viện:

```powershell
uv venv --python 3.14 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường, chạy
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` một lần rồi kích hoạt
lại.

> **Lưu ý:** Dự án chuẩn hóa trên Python 3.14. Đảm bảo môi trường `.venv` đã
> được kích hoạt trước khi chạy các lệnh kế tiếp, hoặc dùng đường dẫn trực tiếp
> `.\.venv\Scripts\python.exe`.

## Huấn luyện

Chạy kiểm toán dữ liệu và pipeline mặc định từ thư mục gốc:

```powershell
python -m src.audit --data-dir data/processed/dataset
python -m src.train
```

Pipeline resize ảnh về 128×128, trích xuất 90 đặc trưng trong mask, so sánh
9 estimator, chạy ablation và chọn mô hình bằng validation. GridSearchCV dùng
StratifiedKFold 5-fold trên train; test chỉ đánh giá sau khi đã chọn model.
Artifact và báo cáo được lưu trong `models/` và `reports/`. Báo cáo gồm CSV,
confusion matrix, ROC/AUC, learning curve, feature importance, mask mẫu và ảnh
dự đoán sai. Để bỏ qua GridSearch:

```powershell
python -m src.train --no-gridsearch
```

Pipeline train sử dụng dataset YOLO Apple/Banana/Mango đã xuất thành ảnh nguyên
cảnh, với artifact riêng:

```powershell
python -m src.dataset --source-dir data/raw --output-dir data/processed/whole-scene
python -m src.train
```

Không dùng tập test để chọn mô hình hoặc điều chỉnh tham số. Hàm tiện ích
`src.predict.predict(image_path)` trả về `(nhãn, xác suất)`.

## Chạy Web Application

Để web dùng model ảnh nguyên cảnh YOLO, cấu hình model/report directory trước
khi chạy FastAPI:

```powershell
$env:FRUIT_MODEL_DIR = "models-whole-scene"
$env:FRUIT_REPORTS_DIR = "reports-whole-scene"
.\.venv\Scripts\python.exe -m app.web
```

Web mặc định đọc artifact từ `models/` và `reports/`; để dùng model
classification Cam/Chuối/Xoài theo thư mục, không cần đổi cấu hình; khi đó chỉ
cần chạy:

```powershell
.\.venv\Scripts\python.exe -m app.web
```

Mở `http://127.0.0.1:8000`. API docs có tại `/docs`. Web mới dùng HTML/CSS/JavaScript, không dùng Streamlit để render. Dashboard, lịch sử và thống kê chỉ hiển thị dữ liệu thực từ model hoặc SQLite; ảnh upload không được lưu.

Giao diện Streamlit cũ vẫn được giữ để tham khảo và có thể chạy riêng:

```powershell
streamlit run app/streamlit_app.py
```

## Kiểm thử

Các bài kiểm thử độc lập hoàn toàn với dataset và không yêu cầu nạp mô hình nặng:

```powershell
python -m pytest tests/ -v
```
