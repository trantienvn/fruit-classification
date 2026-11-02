# Phân loại trạng thái trái cây

Dự án phân loại ảnh trái cây thành 9 lớp: Cam, Chuối, Xoài; mỗi loại gồm
Chưa chín (Unripe), Chín (Ripe) và Hỏng (Spoiled). Dataset mặc định là ảnh
được chia thư mục theo lớp tại `data/processed/dataset/`, với các split
train/val/test có sẵn. Pipeline trích xuất đặc trưng ảnh (màu RGB/HSV, kết cấu
LBP & GLCM, hình dạng) và huấn luyện các mô hình machine learning truyền thống.

## Dataset

Dataset classification hiện hành được lưu cục bộ tại `data/processed/dataset/`
và không được commit vào repository. Cấu trúc lớp, số lượng ảnh và ánh xạ nhãn
được mô tả trong [data/README.md](data/README.md). Nguồn/giấy phép của bộ dữ
liệu này chưa được ghi trong repository.

`data/raw/` vẫn lưu một dataset YOLO cũ Apple/Banana/Mango để tham khảo và
tương thích ngược; nó không phải nguồn dữ liệu mặc định để train.

Dataset YOLO cũ **Fruit Ripeness** do Huzaifa công bố trên
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
- Pipeline scikit-learn mặc định không cần GPU và không dùng trọng số pretrained;
  `requirements.txt` cài PyTorch để có thể huấn luyện MLP trên GPU NVIDIA.
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

Dataset classification được đặt tại `data/processed/dataset/`. Chạy pipeline từ thư mục gốc:

```powershell
python -m src.train --data-dir data/processed/dataset --model-dir models
```

Pipeline đọc ảnh theo các thư mục lớp trong ba split có sẵn, resize ảnh về
128×128, trích xuất 51 đặc trưng, so sánh 7 mô hình và đánh giá mô hình tốt
nhất trên tập test. Mặc định pipeline có chạy GridSearch; để bỏ qua bước này:

```powershell
python -m src.train --data-dir data/processed/dataset --model-dir models --no-gridsearch
```

Muốn huấn luyện MLP bằng GPU NVIDIA, cài dependencies bằng `uv pip install -r requirements.txt`
rồi chạy `python -m src.train --device cuda --no-gridsearch --data-dir data/processed/dataset --model-dir models-gpu`.
Requirements khóa PyTorch ở wheel CUDA 12.6 tương thích driver NVIDIA 566.07 của máy.
Máy vẫn cần NVIDIA driver tương thích.
Xem [hướng dẫn huấn luyện](docs/training.md#6-huấn-luyện-tùy-chọn-trên-gpu-cuda)
để biết cách kiểm tra CUDA, nạp artifact và tiếp tục train với `--resume`.

Muốn tiếp tục train một TorchMLP đã lưu, dùng lại đúng thư mục artifact với `--resume`:

```powershell
.\.venv\Scripts\python.exe -m src.train --resume --device cuda --no-gridsearch --data-dir data/processed/dataset --model-dir models-gpu --epochs 30 --batch-size 64
```

`--epochs` là số epoch tối đa chạy thêm; early stopping có thể dừng sớm. Lệnh nạp
trọng số cùng scaler/PCA hiện có, rồi cập nhật artifact trong `--model-dir`. Có thể
đổi `--device cuda` thành `--device cpu` để tiếp tục trên CPU. `--resume` chỉ áp dụng
cho TorchMLP, không dùng cho các model scikit-learn; mỗi lượt resume khởi tạo lại
optimizer Adam.

Không dùng tập test để chọn mô hình hoặc điều chỉnh tham số. Kết quả huấn luyện
và các báo cáo được ghi vào `models/` và `reports/`.

### Huấn luyện trên ảnh toàn cảnh

Dataset YOLO cũ trong `data/raw/` có sẵn các split `train/val/test`. Có thể
tạo bản classification ảnh nguyên cảnh (không crop bounding box), giữ nguyên
split gốc:

```powershell
python -m src.dataset --source-dir data/raw --output-dir data/processed/whole-scene
```

Chỉ ảnh mà mọi bounding box cùng thuộc một class được đưa vào; ảnh thiếu nhãn,
nhãn lỗi, không có đối tượng hoặc có nhiều class bị loại và thống kê được ghi
trong `data/processed/whole-scene/manifest.json`. Sau đó train trên ảnh nguyên
cảnh:

```powershell
python -m src.train --dataset-mode whole-scene --data-dir data/processed/whole-scene --no-gridsearch
```

Chế độ này mặc định lưu model vào `models-whole-scene/` và báo cáo vào
`reports-whole-scene/`, không ghi đè model classification hiện hành trong
`models/`. Chế độ whole-scene chỉ dùng taxonomy Apple/Banana/Mango của dataset
YOLO cũ. Ảnh có nhiều class trong cùng scene bị loại.

## Chạy Web Application

Sau khi có model trong `models/`, chạy FastAPI từ thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -m app.web
```

Để web dùng model được train trên ảnh toàn cảnh, trỏ biến môi trường tới artifact
đó trước khi chạy:

```powershell
$env:FRUIT_MODEL_DIR = "models-whole-scene"
$env:FRUIT_REPORTS_DIR = "reports-whole-scene"
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
