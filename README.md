# Phân loại độ chín của trái cây

Dự án phân loại ảnh trái cây thành 9 lớp kết hợp loại trái cây và độ chín:
Táo, Chuối, Xoài; mỗi loại có các mức Chưa chín (Unripe), Chín (Ripe) và Quá
chín (Overripe). Dataset đầu vào là ảnh phát hiện đối tượng theo định dạng
YOLO; pipeline cắt từng bounding box thành ảnh mẫu, trích xuất đặc trưng ảnh
(Màu sắc RGB/HSV, Kết cấu LBP & GLCM, Hình dạng) và huấn luyện các mô hình
machine learning truyền thống.

## Dataset

Dataset **Fruit Ripeness** do Huzaifa công bố trên
[Roboflow Universe](https://universe.roboflow.com/huzaifa-qgpco/fruit-ripeness-unjex-xsudt),
sử dụng theo giấy phép
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Giấy phép cho phép sử dụng, chia sẻ và chỉnh sửa, kể cả cho mục đích thương
mại, với điều kiện ghi công phù hợp, dẫn nguồn, liên kết giấy phép và nêu rõ
các thay đổi.

Bản dữ liệu trong dự án được lọc về 9 lớp mục tiêu: các lớp `Rotten` và các
loại trái cây ngoài Táo, Chuối, Xoài không nằm trong ánh xạ đích. Khi sử dụng
lại hoặc phân phối dữ liệu/bản phái sinh, hãy ghi công tác giả Huzaifa, dẫn
[nguồn dataset](https://universe.roboflow.com/huzaifa-qgpco/fruit-ripeness-unjex-xsudt),
liên kết [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) và nêu các
thay đổi này cùng mọi thay đổi khác.

Dataset hiện có 1.559 ảnh train, 445 validation và 223 test. Cấu trúc, nhãn,
ánh xạ lớp và số lượng mẫu được mô tả trong [data/README.md](data/README.md).

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

Dataset đã được đặt tại `data/raw/`. Chạy pipeline từ thư mục gốc:

```powershell
python -m src.train --data-dir data/raw --model-dir models
```

Pipeline đọc ba split YOLO, cắt ảnh theo bounding box, trích xuất 51 đặc trưng,
so sánh 7 mô hình (KNN, Naive Bayes, Logistic Regression, SVM, Decision Tree,
Random Forest, Gradient Boosting) và đánh giá mô hình tốt nhất trên tập test.
Mặc định pipeline có chạy GridSearch; để bỏ qua bước này để huấn luyện nhanh:

```powershell
python -m src.train --data-dir data/raw --model-dir models --no-gridsearch
```

Muốn huấn luyện MLP bằng GPU NVIDIA, cài dependencies bằng `uv pip install -r requirements.txt`
rồi chạy `python -m src.train --device cuda --no-gridsearch --model-dir models-gpu`.
Requirements khóa PyTorch ở wheel CUDA 12.6 tương thích driver NVIDIA 566.07 của máy.
Máy vẫn cần NVIDIA driver tương thích.
Xem [hướng dẫn huấn luyện](docs/training.md#6-huấn-luyện-tùy-chọn-trên-gpu-cuda)
để biết cách kiểm tra CUDA và nạp artifact.

Không dùng tập test để chọn mô hình hoặc điều chỉnh tham số. Kết quả huấn luyện
và các báo cáo được ghi vào `models/` và `reports/`.

## Chạy Web Application

Sau khi có model trong `models/`, chạy FastAPI từ thư mục gốc:

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
