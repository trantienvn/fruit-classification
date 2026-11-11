# Phân loại trạng thái trái cây

Dự án phân loại ảnh trái cây thành 9 lớp. Pipeline mặc định dùng dataset
Cam/Chuối/Xoài × Unripe/Ripe/Spoiled tại `data/processed/dataset/`. Pipeline
chuyển đổi ảnh 64×64 RGB sang vector số chuẩn hóa `[0, 1]` (12.288 chiều,
không trích xuất đặc trưng thủ công), giảm chiều bằng PCA và huấn luyện các
mô hình machine learning truyền thống bằng scikit-learn trên CPU.

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

## Tiền xử lý dữ liệu

Chuyển đổi toàn bộ ảnh từ `data/processed/dataset/` sang ma trận số chuẩn hóa
`[0, 1]` (64×64 RGB = 12.288 chiều, lưu tại `data/processed/numeric/numeric_dataset.npz`):

```powershell
python -m src.preprocess
```

Dùng `--force` để xử lý lại khi cần, hoặc `--data-dir` để chọn dataset khác:

```powershell
python -m src.preprocess --force
```

Mỗi lần train (`python -m src.train`), chương trình tự động nạp bộ dữ liệu số
`.npz` đã qua xử lý. Nếu chưa có file `.npz` hoặc dataset ảnh thay đổi,
`train` sẽ tự động chạy tiền xử lý trước rồi huấn luyện từ dữ liệu số đó.

## Huấn luyện

Chạy kiểm toán dữ liệu và pipeline mặc định từ thư mục gốc:

```powershell
python -m src.audit --data-dir data/processed/dataset
python -m src.preprocess
python -m src.train
```

Pipeline chuẩn hóa pixel với `StandardScaler`, giảm chiều bằng PCA (95% phương
sai), so sánh 9 estimator và chọn mô hình bằng validation. GridSearchCV dùng
StratifiedKFold 5-fold trên train; test chỉ đánh giá sau khi đã chọn model.
Artifact và báo cáo được lưu trong `models/` và `reports/`. Báo cáo gồm CSV,
confusion matrix, ROC/AUC, learning curve, feature importance và ảnh dự đoán
sai. Để bỏ qua GridSearch:

```powershell
python -m src.train --no-gridsearch
```

Để chạy trên dataset YOLO Apple/Banana/Mango đã xuất thành ảnh nguyên cảnh
(`models-whole-scene/` và `reports-whole-scene/`):

```powershell
python -m src.dataset --source-dir data/raw --output-dir data/processed/whole-scene
python -m src.preprocess --data-dir data/processed/whole-scene
python -m src.train --data-dir data/processed/whole-scene
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
