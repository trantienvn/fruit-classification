# Hướng dẫn cài đặt môi trường

Tài liệu hướng dẫn thiết lập môi trường phát triển cho đồ án **Phân loại chất lượng và độ chín trái cây**.

---

## 1. Yêu cầu hệ thống

- **Hệ điều hành:** Windows 10 hoặc Windows 11 (64-bit).
- **Python:** Phiên bản `3.14.x` (chuẩn hóa dự án: Python 3.14.7).
- **Công cụ quản lý gói:** [`uv`](https://github.com/astral-sh/uv) (khuyến nghị) hoặc `pip`.
- **Phần cứng:** CPU đa nhân; GPU NVIDIA chỉ cần cho backend CUDA tùy chọn.

`requirements.txt` cài PyTorch CUDA 12.6 để hỗ trợ huấn luyện GPU. GPU NVIDIA
và driver tương thích vẫn cần có sẵn trên máy nếu muốn dùng CUDA.
Không cần GPU — pipeline hoạt động hoàn toàn trên CPU.

---

## 2. Cài đặt công cụ `uv`

Nếu máy chưa cài `uv`, mở PowerShell và chạy:

```powershell
winget install astral-sh.uv
```

Sau khi cài đặt xong, khởi động lại PowerShell để nhận lệnh `uv`.

---

## 3. Khởi tạo môi trường ảo và cài đặt thư viện

Tại thư mục gốc của dự án (`fruit-classification/`), thực hiện:

### Bước 1: Tạo môi trường ảo với Python 3.14

```powershell
uv venv --python 3.14 .venv
```

### Bước 2: Kích hoạt môi trường ảo

```powershell
.\.venv\Scripts\Activate.ps1
```

*Lưu ý:* Nếu PowerShell báo lỗi Execution Policy, hãy mở PowerShell quyền Administrator hoặc chạy lệnh sau với quyền người dùng hiện tại:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Bước 3: Cài đặt các phụ thuộc từ `requirements.txt`

```powershell
uv pip install -r requirements.txt
```

Danh mục các thư viện chính:
- `numpy`, `scipy`: Xử lý số học và ma trận.
- `scikit-learn`: Classifier trên embedding CNN và các công cụ tiền xử lý, đánh giá.
- `torch`, `torchvision`: CNN backbone và huấn luyện MLP (GPU CUDA hoặc CPU).
- `Pillow`: Đọc, cắt và tiền xử lý ảnh.
- `matplotlib`, `seaborn`: Trực quan hóa và xuất biểu đồ thực nghiệm.
- `joblib`: Đóng gói và tuần tự hóa mô hình.
- `pytest`: Kiểm thử tự động.

---

## 4. Kiểm tra cài đặt

Kiểm tra bộ test độc lập để xác nhận môi trường đã sẵn sàng:

```powershell
python -m pytest tests/ -v
```

Kết quả mong đợi: `17 passed`.
