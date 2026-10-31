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

---

### Issue #5: Source chưa đạt kiểm tra lint/style của Ruff
- **Phát hiện (2026-10-31)**: Chạy Ruff với các nhóm quy tắc `E,F,I` trên `src/` có 67 diagnostics: 59 dòng vượt giới hạn 88 ký tự (`E501`), 6 import không dùng (`F401`) và 2 import block sai thứ tự (`I001`). Ruff formatter báo 8 module cần định dạng lại.
- **Ảnh hưởng**: Vấn đề định dạng mã nguồn; không ảnh hưởng runtime hoặc kiểm thử.

---

### Issue #6: Lỗi `UnicodeEncodeError: 'charmap'` khi in tiếng Việt trên console Windows
- **Hiện tượng**: Lệnh `python -m src.train` văng lỗi:
  `UnicodeEncodeError: 'charmap' codec can't encode character '\u1ea4' in position 11: character maps to <undefined>`.
- **Nguyên nhân**: PowerShell/CMD trên Windows mặc định mã hóa ngõ ra theo mã trang ANSI `cp1252` thay vì UTF-8, nên các ký tự tiếng Việt có dấu trong `print()` không thể mã hóa được.
- **Giải pháp**:
  - Bổ sung cấu hình tự động nhận diện và tái cấu hình luồng xuất chuẩn tại đầu các file chạy chính:
    ```python
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ```
  - Thiết lập biến môi trường `$env:PYTHONUTF8="1"` khi chạy lệnh PowerShell.

---

### Issue #7: Lỗi `with_suffix` khi đọc nhãn Roboflow có nhiều dấu chấm trong tên file
- **Hiện tượng**: `load_split` tải được 0 mẫu đối tượng trên dataset thật (`0 mẫu — 0.0s`), gây ra lỗi `ValueError: Expected 2D array, got 1D array instead: array=[]`.
- **Nguyên nhân**: Tên file xuất từ Roboflow Universe có định dạng `tên_ảnh.rf.UUID.jpg` (ví dụ: `-1_jpg.rf.56ecaa8de6a86784bf5b0e11af267026.jpg`). Trong Python `pathlib`, `img_path.stem` là `-1_jpg.rf.56ecaa8de6a86784bf5b0e11af267026`. Khi gọi `(label_dir / img_path.stem).with_suffix(".txt")`, phương thức `with_suffix` coi chuỗi sau dấu chấm cuối cùng (`.56ecaa8...`) là phần mở rộng cũ và thay thế nó bằng `.txt`, tạo thành `label_dir / -1_jpg.rf.txt` (sai lệch hoàn toàn so với tệp nhãn thật).
- **Giải pháp**:
  - Thay đổi cách nối đuôi tệp nhãn: `label_path = label_dir / f"{img_path.stem}.txt"`.
  - Sau khi sửa, toàn bộ 2.507 mẫu gán nhãn được nạp chính xác 100%.

---

### Issue #8: Lệch kích thước mẫu trong hàm vẽ biểu đồ ROC (`full_report`)
- **Hiện tượng**: `full_report` nhận `y_true` là nhãn test (246 mẫu) nhưng lại nhận ma trận đặc trưng tập val `X_val` (491 mẫu), dẫn đến nguy cơ lỗi `ValueError: Found input variables with inconsistent numbers of samples`.
- **Giải pháp**: Cập nhật hàm `full_report` nhận tham số `X_test=X_te` khớp với `y_test`, đảm bảo kích thước đồng nhất khi vẽ 9 đường cong ROC One-vs-Rest.

---

### Issue #9: Lỗi thuộc tính `gs.best_params__` trong GridSearchCV
- **Hiện tượng**: Thuộc tính của `GridSearchCV` bị gõ nhầm thành 2 dấu gạch dưới `best_params__` thay vì 1 dấu gạch dưới `best_params_`.
- **Giải pháp**: Sửa lại thành `gs.best_params_` và đảm bảo chỉ thay thế mô hình tốt nhất nếu điểm F1 của mô hình tinh chỉnh thực sự vượt qua mô hình ban đầu.

### Issue #10: Không tìm thấy package `src` khi chạy Streamlit từ thư mục `app/`
- **Hiện tượng**: Chạy `streamlit run streamlit_app.py` sau khi chuyển vào `app/` gây `ModuleNotFoundError: No module named 'src'`.
- **Nguyên nhân**: Thư mục làm việc hiện tại là `app/`, nên thư mục gốc repository không có trong đường dẫn import; các đường dẫn tương đối đến `models/` và `reports/` cũng phụ thuộc thư mục khởi chạy.
- **Giải pháp**: Xác định `PROJECT_ROOT` từ `Path(__file__).resolve()`, thêm vào `sys.path` trước khi import `src`, và neo đường dẫn model/báo cáo vào thư mục gốc đó.
- **Xác minh**: Chạy Streamlit AppTest từ thư mục `app/`, xác nhận import thành công, mô hình được nạp và metric test accuracy 93,5% hiển thị.
