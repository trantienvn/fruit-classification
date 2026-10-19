# Phân loại chất lượng và độ chín của trái cây

Dự án khung dùng PyTorch/torchvision và MobileNetV3 pretrained để dự đoán **loại trái cây**, **độ chín** và **chất lượng** từ ảnh. Hiện workspace chưa có dataset hoặc trọng số mô hình; vì vậy chưa thể huấn luyện hay đưa ra dự đoán thực tế. Không có kết quả giả được tạo.

## Yêu cầu

- Windows 10/11, Python 3.11 và `pip`.
- Có kết nối mạng khi cài thư viện. Không cần GPU; để dùng CUDA, cài đúng bản PyTorch theo hướng dẫn chính thức tại <https://pytorch.org/get-started/locally/>.

## Cài đặt trên Windows

Mở PowerShell tại thư mục dự án:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường, chạy `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` một lần rồi kích hoạt lại.

## Chuẩn bị dữ liệu

Đưa ảnh vào `data/raw/` theo quy ước thư mục được ghi trong [data/README.md](data/README.md). Mỗi ảnh cần có đủ ba nhãn. Chia tập train/validation/test theo nguồn thu hoặc mẫu vật để tránh rò rỉ ảnh gần trùng giữa các tập. Không đưa dữ liệu riêng tư hoặc ảnh không được phép sử dụng.

## Lệnh dự kiến

```powershell
# Kiểm tra khung và các bài kiểm thử không cần dataset
python -m pytest

# Giai đoạn dữ liệu/mô hình chưa được triển khai trong bộ khung này.
# Sau khi có dữ liệu thực tế, tiếp tục triển khai và chạy pipeline train/evaluate.
streamlit run app/streamlit_app.py
```

Streamlit hiện chỉ kiểm tra sự có mặt của checkpoint và hướng dẫn chuẩn bị dữ liệu; chưa có checkpoint thì không hiển thị dự đoán.

## Thiết kế pipeline

MobileNetV3 pretrained làm backbone ảnh. Với dữ liệu đủ nhãn, có thể dùng một backbone và ba đầu ra phân loại (loại trái cây, độ chín, chất lượng), tối ưu tổng loss có trọng số; nếu nhãn bị thiếu hoặc phân phối/tập dữ liệu khác nhau, huấn luyện ba bộ phân loại riêng sẽ phù hợp hơn. Quyết định cuối cùng cần dựa trên thống kê dữ liệu và đánh giá trên tập test độc lập. `src/data.py`, `train.py`, `evaluate.py`, `predict.py` là các điểm mở rộng; giai đoạn khung chưa tải trọng số hay huấn luyện.

## Trạng thái

- Dataset: **chưa có** (`data/raw/` trống).
- Checkpoint: **chưa có** (`models/` trống).
- Huấn luyện/đánh giá: chưa chạy; cần dữ liệu thật trước.
