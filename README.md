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
- Không bắt buộc GPU hoặc tải trọng số pretrained.
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

Không dùng tập test để chọn mô hình hoặc điều chỉnh tham số. Kết quả huấn luyện
và các báo cáo được ghi vào `models/` và `reports/`.

## Chạy giao diện

Sau khi huấn luyện và tạo checkpoint:

```powershell
streamlit run app/streamlit_app.py
```

Ứng dụng cung cấp 3 tab:
1. **Dự đoán**: Tải ảnh (hoặc kéo thả), crop trái cây hoặc dự đoán toàn ảnh, hiển thị nhãn và phân bố xác suất các lớp.
2. **Lịch sử**: Xem lại các lượt dự đoán trước đó kèm ảnh thu nhỏ và mốc thời gian.
3. **Thông tin mô hình**: Xem cấu hình mô hình hiện tại, các chỉ số đánh giá thực tế và các biểu đồ thực nghiệm (Confusion Matrix, ROC Curves, Model Comparison).

## Kiểm thử

Các bài kiểm thử độc lập hoàn toàn với dataset và không yêu cầu nạp mô hình nặng:

```powershell
python -m pytest tests/ -v
```
