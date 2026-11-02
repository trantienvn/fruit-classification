# Quyết định kỹ thuật đã thống nhất (Technical Decisions)

Tài liệu này ghi lại các quyết định kỹ thuật cốt lõi đã được thống nhất giữa Người dùng (User) và AI Agent, làm căn cứ bắt buộc cho mọi phát triển tiếp theo.

---

## 1. Quyết định môi trường thực thi (Q1)
- **Quyết định**: Đồng bộ môi trường sang phiên bản **Python 3.14** (cụ thể: Python 3.14.7).
- **Lý do**: Môi trường hiện hữu của người dùng đang sử dụng Python 3.14 (thể hiện qua bytecode `cpython-314.pyc` và Python core cài đặt trên máy).
- **Thực thi**: Tạo môi trường ảo `.venv` bằng công cụ `uv` liên kết tới `C:\Users\TranTien\AppData\Local\Python\pythoncore-3.14-64\python.exe`.
- **Phương án bị loại bỏ**: Ép cài lại Python 3.11 gây xung đột với binary có sẵn trên máy người dùng.

---

## 2. Quyết định phạm vi bài toán và cấu trúc nhãn (Q2, DEPRECATED 2026-11-02)
- **Quyết định**: Giải quyết bài toán **Phân loại 9 lớp kết hợp (Loại trái cây × Mức độ chín)**.
  - Danh sách 9 nhãn: `Apple Unripe`, `Apple Ripe`, `Apple Overripe`, `Banana Unripe`, `Banana Ripe`, `Banana Overripe`, `Mango Unripe`, `Mango Ripe`, `Mango Overripe`.
- **Trạng thái hiện tại**: Quyết định này chỉ còn áp dụng cho dữ liệu YOLO cũ trong `data/raw/`; taxonomy mặc định đã được thay thế bởi Q9.
- **Lý do ban đầu**:
  - Dữ liệu thực tế trong `data/raw/data.yaml` và nhãn YOLO chỉ chứa 9 lớp kết hợp trên.
  - Không có dữ liệu phân hạng chất lượng thương mại độc lập (Class 1, Class 2, Extra Class).
  - Không tự bịa đặt nhãn hoặc giả lập phân loại khi dataset không hỗ trợ.
- **Phương án bị loại bỏ**:
  - Tách thành 2 mô hình độc lập (1 nhận diện quả, 1 nhận diện độ chín): Không cần thiết vì số lớp tổng hợp nhỏ (9 lớp) và nhãn gộp đã gắn chặt theo từng loại quả.
  - Tự thêm nhãn chất lượng (Extra/Class 1): Vi phạm nguyên tắc bảo toàn tính chân thực của dữ liệu học thuật.

---

## 3. Quyết định tiền xử lý dữ liệu từ YOLO sang Classification (Q3, legacy)
- **Quyết định**: **Crop ảnh theo Bounding Box**.
- **Lý do**: Dataset hiện tại được gắn nhãn theo định dạng YOLO Object Detection. Việc crop theo bounding box loại bỏ tối đa tạp chất hậu cảnh, tập trung toàn bộ pixel vào đối tượng trái cây trước khi trích xuất đặc trưng.
- **Thực thi**:
  - Tọa độ YOLO `(cx, cy, bw, bh)` được chuyển thành `(x1, y1, x2, y2)` pixel.
  - Crop và resize về kích thước chuẩn **128×128** bằng thuật toán `PIL.Image.LANCZOS`.
- **Phương án bị loại bỏ**:
  - Dùng toàn bộ ảnh thô + Otsu: Dễ bị nhiễu do nền phức tạp (cây cối, bàn tay, quầy hàng).

---

## 4. Quyết định loại bỏ Deep Learning khỏi `requirements.txt`
- **Quyết định**: Loại bỏ hoàn toàn `torch` và `torchvision`, chuyển sang stack Học máy truyền thống (`scikit-learn`, `scikit-image`, `opencv-python-headless`, `scipy`).
- **Lý do**:
  - Đề tài học phần Học máy của nhóm 03 định hướng nghiên cứu thuật toán học máy truyền thống kết hợp trích xuất đặc trưng thủ công (Feature Engineering: Color, GLCM, LBP, Shape).
  - Tránh tải thư viện nặng (PyTorch nặng hàng GB) vi phạm quy tắc trong `AGENTS.md`.

---

## 5. Quyết định kiến trúc giao diện người dùng (DEPRECATED ngày 2026-10-31)
- **Quyết định**: Sử dụng **Streamlit** làm Web Dashboard tích hợp cả Frontend & Backend.
- **Lý do**:
  - Phù hợp với đồ án học thuật/Data Science: Dễ triển khai, demo trực tiếp trên laptop, có sẵn cơ chế cache model (`@st.cache_resource`).
  - Tích hợp phong cách giao diện tự thiết kế theo `frontend-design` (màu sắc chủ đạo xanh lá sẫm `#2D6A4F`, xanh lá tươi `#52B788`, nền kem `#FAF7F2`, typography rõ ràng, không dùng template SaaS đại trà).
  - Cung cấp đầy đủ: Upload & dự đoán, độ tin cậy theo lớp, lịch sử dự đoán cục bộ và tab thông tin mô hình kèm biểu đồ thực nghiệm.

---

## 6. Chuyển Streamlit sang Web Application (Q6, CONFIRMED 2026-10-31)
- **Quyết định**: Dùng FastAPI làm backend, HTML5/CSS3/JavaScript ES6+ làm frontend; phục vụ giao diện và REST API cùng origin. FastAPI/Uvicorn nạp `FruitClassifier` một lần trong startup.
- **Lý do**: Đáp ứng yêu cầu tách UI khỏi Streamlit trong khi vẫn dùng pipeline suy luận Python đã có.
- **Bảo toàn pipeline**: Không sửa `src/`, model, scaler, PCA, đặc trưng, thứ tự 51 feature hoặc nhãn 9 lớp. Không huấn luyện lại.
- **Ảnh và lịch sử**: Ảnh chỉ được dùng khi suy luận; SQLite lưu metadata dự đoán, không lưu ảnh. Thống kê người dùng lấy từ lịch sử này, còn metrics đánh giá lấy từ `models/config.json`.
- **Xác suất**: API chỉ gửi `confidence` và `probabilities` khi classifier có `predict_proba`; UI có trạng thái khi trường này vắng mặt.
- **Tương thích**: Giữ `app/streamlit_app.py` và dependency Streamlit để chạy bản cũ khi cần đối chiếu.

## 7. Backend huấn luyện GPU tùy chọn (Q7, CONFIRMED 2026-10-31)
- **Quyết định**: Giữ pipeline scikit-learn bảy mô hình làm mặc định; bổ sung `--device cuda` để huấn luyện `TorchMLP` bằng PyTorch trên GPU NVIDIA.
- **Phạm vi**: MLP dùng các vector 51 đặc trưng hiện tại. Đọc/crop ảnh, trích xuất đặc trưng, StandardScaler và PCA vẫn chạy trên CPU. Chế độ CUDA không chạy GridSearchCV.
- **Dependency**: Khai báo `torch==2.14.1+cu126` trong `requirements.txt` theo yêu cầu mới và driver NVIDIA 566.07; CUDA runtime đi kèm wheel. Driver NVIDIA vẫn được cài riêng trên Windows.
- **Tương thích artifact**: Lưu trọng số dưới dạng NumPy; artifact GPU có thể suy luận CPU nếu CUDA không khả dụng, nhưng môi trường suy luận vẫn cần PyTorch.
- **Quan hệ với Q4**: Q4 tiếp tục áp dụng cho pipeline mặc định; yêu cầu mới cho phép thêm backend Deep Learning tùy chọn và cài PyTorch cùng dependencies.

## 8. Chế độ huấn luyện ảnh toàn cảnh (Q8, CONFIRMED 2026-11-02)
- **Quyết định**: Bổ sung chế độ classification trên ảnh nguyên cảnh bên cạnh chế độ YOLO crop hiện có, chọn bằng `--dataset-mode whole-scene`.
- **Tách dữ liệu**: Giữ nguyên train/val/test YOLO để không trộn split. Chỉ giữ ảnh mà mọi bounding box cùng thuộc một class; loại ảnh thiếu/nhãn lỗi/đa class và ghi nguyên nhân vào manifest.
- **Tiền xử lý**: Ảnh toàn cảnh được resize về 128×128, khớp với inference hiện tại; không crop bbox.
- **Artifact**: Mặc định tách vào `models-whole-scene/` và `reports-whole-scene/`; không ghi đè model crop. Resume không cho phép đổi chế độ dữ liệu.
- **Giới hạn**: Model vẫn là classifier một nhãn cho toàn ảnh; không hỗ trợ ảnh có nhiều loại quả khác nhãn trong cùng cảnh.

## 9. Dataset classification Cam/Chuối/Xoài (Q9, CONFIRMED 2026-11-02)
- **Quyết định**: Dùng `data/processed/dataset/{train,val,test}/` làm dataset classification mặc định; không đổi nội dung `data/raw/`.
- **Ánh xạ lớp**: Giữ 9 class theo thứ tự Orange/Banana/Mango × Unripe/Ripe/Spoiled; các thư mục `*-hong` được xác nhận là `Spoiled`, không gộp thành `Overripe`.
- **Huấn luyện**: Dùng mode `folder`, chuẩn hóa ảnh 128×128; đánh giá validation/test theo split có sẵn. Model/report mới thay artifact mặc định trong các thư mục local bị Git ignore.
- **Tương thích**: Các mode `yolo` và `whole-scene` tiếp tục dùng taxonomy cũ Apple/Banana/Mango × Unripe/Ripe/Overripe; class names của model suy luận được đọc từ config artifact.
- **Nguồn dữ liệu**: Bộ thư mục classification hiện có tại máy, không được commit. Nguồn và giấy phép chưa được ghi trong repository, không suy đoán nguồn hoặc giấy phép.
