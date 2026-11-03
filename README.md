# Phân loại trạng thái trái cây

Dự án phân loại ảnh trái cây thành 9 lớp. Pipeline mặc định dùng ảnh nguyên
cảnh (không crop bounding box) từ `data/processed/whole-scene/`, giữ các split
train/val/test và taxonomy Apple/Banana/Mango của dataset YOLO cũ. Có thể chọn
dataset classification Cam/Chuối/Xoài ở chế độ `folder`. Pipeline trích xuất
đặc trưng ảnh (màu RGB/HSV, kết cấu LBP & GLCM, hình dạng) và chỉ huấn luyện
các mô hình machine learning truyền thống trên CPU.

## Dataset

Dataset ảnh nguyên cảnh mặc định được tạo cục bộ từ `data/raw/` vào
`data/processed/whole-scene/`; không crop trái cây và loại ảnh nhiều class vì
không thể gán một nhãn cho toàn cảnh. Dữ liệu sinh ra không được commit vào
repository.

Dataset classification Cam/Chuối/Xoài tại `data/processed/dataset/` là lựa
chọn thêm, không phải chế độ train mặc định. Cấu trúc lớp, số lượng ảnh và ánh
xạ nhãn được mô tả trong [data/README.md](data/README.md). Nguồn/giấy phép của
bộ dữ liệu này chưa được ghi trong repository.

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

Tạo dataset ảnh nguyên cảnh một lần từ các split YOLO trong `data/raw/`:

```powershell
python -m src.dataset --source-dir data/raw --output-dir data/processed/whole-scene
```

Sau đó chạy pipeline mặc định từ thư mục gốc:

```powershell
python -m src.train
```

Pipeline mặc định dùng `data/processed/whole-scene/`, resize toàn ảnh về
128×128, trích xuất 51 đặc trưng, so sánh 7 mô hình và đánh giá mô hình tốt
nhất trên tập test. Artifact và báo cáo được lưu trong `models-whole-scene/`
và `reports-whole-scene/`. Mặc định pipeline có chạy GridSearch; để bỏ qua:

```powershell
python -m src.train --no-gridsearch
```

Để dùng dataset classification Cam/Chuối/Xoài theo thư mục, chọn rõ mode và
đường dẫn dữ liệu/model:

```powershell
python -m src.train --dataset-mode folder --data-dir data/processed/dataset --model-dir models
```

Không dùng tập test để chọn mô hình hoặc điều chỉnh tham số. Với mode `folder`,
artifact và báo cáo được ghi vào `models/` và `reports/`; mode whole-scene dùng
các thư mục riêng như trên.

## Chạy Web Application

Để web dùng model ảnh nguyên cảnh được train mặc định, cấu hình model/report
directory trước khi chạy FastAPI:

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
