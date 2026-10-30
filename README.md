# Phân loại độ chín của trái cây

Dự án phân loại ảnh trái cây thành 9 lớp kết hợp loại trái cây và độ chín:
Táo, Chuối, Xoài; mỗi loại có các mức Chưa chín (Unripe), Chín (Ripe) và Quá
chín (Overripe). Dataset đầu vào là ảnh phát hiện đối tượng theo định dạng
YOLO; pipeline cắt từng bounding box thành ảnh mẫu, trích xuất đặc trưng ảnh
và huấn luyện các mô hình machine learning truyền thống.

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

- Windows 10/11, Python 3.14 và `pip`.
- Không bắt buộc GPU hoặc tải trọng số pretrained.
- Có kết nối mạng để cài các thư viện trong `requirements.txt`.

## Cài đặt trên Windows

Mở PowerShell tại thư mục dự án:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường, chạy
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` một lần rồi kích hoạt
lại.

## Huấn luyện

Dataset đã được đặt tại `data/raw/`. Chạy pipeline từ thư mục gốc:

```powershell
python -m src.train --data-dir data/raw --model-dir models
```

Pipeline đọc ba split YOLO, cắt ảnh theo bounding box, trích xuất đặc trưng,
so sánh các mô hình và đánh giá mô hình đã chọn trên tập test. Mặc định pipeline
có chạy GridSearch; để bỏ qua bước này:

```powershell
python -m src.train --data-dir data/raw --model-dir models --no-gridsearch
```

Không dùng tập test để chọn mô hình hoặc điều chỉnh tham số. Kết quả huấn luyện
và các báo cáo được ghi vào `models/` và `reports/`; hiện hai thư mục này chưa
có checkpoint hay kết quả huấn luyện.

## Chạy giao diện

Sau khi huấn luyện và tạo checkpoint:

```powershell
streamlit run app/streamlit_app.py
```

Ứng dụng cho phép tải ảnh để dự đoán lớp trái cây/độ chín và xem lịch sử dự
đoán. Nếu chưa có checkpoint trong `models/`, cần chạy huấn luyện trước.

## Kiểm thử

Các bài kiểm thử không yêu cầu tải dataset hoặc mô hình:

```powershell
python -m pytest
```
