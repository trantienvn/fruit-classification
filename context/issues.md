# Nhật ký xử lý sự cố & Kinh nghiệm (Issues & Learnings)

## 1. Danh sách sự cố kỹ thuật

### Issue #1: Quyền truy cập Python 3.14 bị chặn trong Sandbox Windows
- **Hiện tượng**: Khi gọi `python --version` hoặc chạy lệnh qua đường dẫn `C:\Users\TranTien\AppData\Local\Python\bin\python.exe` trong sandbox mặc định, Windows trả về lỗi: `The file cannot be accessed by the system` hoặc `Access is denied` (os error 1920 / UnauthorizedAccessException).
- **Nguyên nhân**: Môi trường sandbox phân quyền nghiêm ngặt, không cho phép truy cập trực tiếp vào thư mục cài đặt gốc của người dùng tại `AppData\Local\Python\pythoncore-3.14-64`.
- **Giải pháp**:
  - Dùng `uv venv --python 3.14 .venv` với `BypassSandbox: true` để tạo virtual environment nội bộ nằm ngay trong workspace (`d:\code\Python\fruit-classification\.venv`).
  - Khi chạy các lệnh thực thi nặng hoặc kiểm thử cần liên kết tới runtime gốc, kích hoạt quyền `BypassSandbox: true`.

---

### Issue #2: Lỗi DNS/Network khi uv tải thư viện bên trong sandbox
- **Hiện tượng**: `uv` báo lỗi `dns error: No such host is known (os error 11001)` khi cố gắng tải Python hoặc các gói wheel từ PyPI.
- **Nguyên nhân**: Sandbox mặc định không có kết nối Internet để đảm bảo an toàn.
- **Giải pháp**:
  - Phân tách lệnh: Lệnh cần mạng (cài package qua `uv pip install`) được chạy với `BypassSandbox: true`.
  - Toàn bộ các thao tác xử lý mã nguồn, viết file, kiểm tra dữ liệu chạy bình thường trong sandbox.

---

### Issue #3: Sai lệch tọa độ bounding box do làm tròn số thực trong `test_parse_yolo_box_valid`
- **Hiện tượng**: Unit test `test_parse_yolo_box_valid` bị fail: `assert (9 == 10)`.
- **Nguyên nhân**:
  - Trong phép tính: `(cx - bw / 2) * img_w` với `cx=0.5, bw=0.8, img_w=100`:
  - Trong biểu thức số thực chuẩn IEEE 754: `0.5 - 0.4 = 0.09999999999999998`.
  - Phép nhân `0.09999999999999998 * 100 = 9.999999999999998`.
  - Khi ép kiểu bằng `int()`, phần thập phân bị chặt đứt thành `9` thay vì `10`.
- **Giải pháp khắc phục**:
  - Sử dụng hàm `round(...)` trước khi ép kiểu `int`: `int(round((cx - bw / 2) * img_w))` cho cả `x1, y1, x2, y2` trong `src/dataset.py`.
- **Phòng ngừa**: Luôn sử dụng `round()` khi chuyển đổi tọa độ tương đối (0.0 - 1.0) sang tọa độ pixel nguyên.

---

### Issue #4: Mâu thuẫn yêu cầu công nghệ (PyTorch vs Học máy truyền thống)
- **Hiện tượng**: Tệp `requirements.txt` ban đầu và `README.md` cũ đề cập `torch`, `torchvision` và mô hình `MobileNetV3`, trong khi đề tài đồ án học phần Học máy của nhóm 03 yêu cầu phương pháp học máy truyền thống kết hợp trích xuất đặc trưng ảnh thủ công.
- **Nguyên nhân**: Template cũ được khởi tạo từ một dự án deep learning khác.
- **Giải pháp**:
  - Dọn dẹp sạch sẽ `requirements.txt`: Bỏ `torch`, thêm `scikit-learn`, `scikit-image`, `opencv-python-headless`, `scipy`.
  - Khảo sát và đối chiếu với `docs/AGENT_PROJECT_SPEC.md` để xác nhận và tuân thủ đúng yêu cầu học phần.

