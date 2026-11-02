# Nhật ký thay đổi (Changelog)

Toàn bộ các thay đổi mã nguồn và tài liệu của dự án được ghi nhận theo chuẩn [Keep a Changelog](https://keepachangelog.com/).

## [2026-11-02]

### Chuyển dataset mặc định sang Cam/Chuối/Xoài
- Thêm loader dataset classification theo thư mục train/val/test; mapping `*-hong` thành `Spoiled`.
- Cập nhật nhãn suy luận, thống kê, Web/Streamlit UI và tài liệu; giữ nguyên dataset YOLO Apple/Banana/Mango ở chế độ tương thích riêng.
- Train lại SVM từ dataset local: test accuracy 91,30%, Macro F1 0,9115; cập nhật báo cáo `reports/` và artifact local trong `models/`.
- Kiểm tra: 30 tests passed; inference smoke trên ảnh cam-hỏng trong test split trả `Orange Spoiled`, 9 lớp/xác suất hợp lệ; không commit dataset hoặc model artifacts.

### Thêm biểu đồ độ quan trọng đặc trưng cho TorchMLP
- Model có tree importance/coef tiếp tục dùng chỉ số tích hợp; model khác dùng permutation importance theo Macro F1 trên validation, tránh dùng test.
- `reports-gpu/feature_importance.png` được tạo cho model TorchMLP đang chạy bằng 491 mẫu validation và được phục vụ tại `/reports/feature_importance.png`; trang model-info đã có sẵn mục hiển thị.
- Bổ sung unit test generation và static report endpoint; `pytest tests/`: 28 passed, 1 warning; compile/`git diff --check` đạt.

### Thêm train trên ảnh toàn cảnh
- Thêm công cụ xuất ảnh nguyên cảnh từ split YOLO sang cấu trúc classification, giữ nguyên train/val/test và ghi manifest thống kê các ảnh bị loại.
- Thêm `--dataset-mode whole-scene`, lưu model/báo cáo riêng; giữ chế độ crop bbox làm mặc định.
- Dataset hiện có giữ 1.547/443/220 ảnh ở train/val/test; RandomForest đạt test accuracy 91,82%, Macro F1 0,9174.
- Kiểm tra: 26 tests passed; inference smoke test với model whole-scene đạt; `git diff --check` và `py_compile` đạt.

### Thay ảnh trực tiếp bằng kéo thả
- Cho phép thả ảnh mới lên vùng xem trước để thay ảnh hiện tại; thêm gợi ý thao tác và trạng thái viền khi kéo ảnh vào.
- Nếu ảnh thay thế vượt quá 10 MB, giữ ảnh hiện tại; bỏ qua kết quả của lượt phân loại cũ nếu người dùng đã đổi ảnh trong lúc xử lý.
- Mở rộng vùng thả ra toàn bộ thẻ ảnh đầu vào, thêm overlay hướng dẫn và hiển thị đầy đủ lớp tiếng Việt ở kết quả chính (ví dụ “Chuối · Quá chín”).
- Đã rà soát diff và chạy `git diff --check`; không chạy test trong phiên này.

### Trả tên lớp tiếng Việt từ Web API
- Thêm `class_name_vi` vào response dự đoán và lịch sử; giữ nguyên `class_name` canonical từ model.
- Lịch sử cũ cũng được dịch khi serialize, không cần migrate SQLite; trang nhận diện ưu tiên field tiếng Việt do API trả về.
- Sinh lại `app/web/openapi.json` từ schema FastAPI; không chạy test trong phiên này.

### Tiếp tục huấn luyện TorchMLP
- Thêm `--resume` để warm-start TorchMLP từ `classifier.joblib`, dùng lại scaler/PCA đã lưu và cho phép chạy thêm số epoch chỉ định.
- Ghi nhận tổng epoch, epoch của lượt vừa chạy và trạng thái resume trong cấu hình model.
- Cập nhật hướng dẫn huấn luyện, README và context.
- Xác minh sau triển khai: 24 tests đạt; smoke warm-start CPU qua serialize/deserialize đạt; artifact resume trên CUDA nạp và dự đoán thành công.

## [2026-10-31]

### Thêm backend huấn luyện CUDA tùy chọn
- Thêm `src/torch_model.py` với MLP PyTorch, early stopping theo validation loss và lưu trọng số dạng NumPy.
- Thêm `--device cuda`, `--epochs`, `--batch-size`; pipeline scikit-learn CPU vẫn là mặc định.
- Thêm `torch==2.14.1+cu126` vào `requirements.txt` để khớp driver NVIDIA 566.07; driver NVIDIA cài riêng.
- Ghi hướng dẫn kiểm tra GPU, chọn bản CUDA trong PyTorch selector nếu cần và nạp artifact.
- Tách báo cáo CUDA vào `reports-gpu/` để không ghi đè báo cáo CPU.
- Chưa chạy test hoặc huấn luyện CUDA trong phiên này.

### Chuyển sang Web Application HTML/CSS/JavaScript
- Thêm `app/web/`: FastAPI routers/services, validation upload, model startup một lần, static frontend và OpenAPI snapshot.
- Thêm năm màn hình responsive cho tổng quan, nhận diện, lịch sử, thống kê và model info; chart lấy từ API history/config thật.
- Thêm SQLite metadata history; không lưu ảnh, không tạo số liệu dự đoán minh họa.
- Bổ sung FastAPI/Uvicorn/multipart và HTTP test dependencies trong `requirements.txt`; bỏ qua database runtime trong Git.
- Thêm `tests/test_web_api.py`, `tests/conftest.py`; tổng kiểm thử 24/24 đạt.
- Cập nhật README, `docs/inference.md`, context architecture/decisions/progress/API contract và task board.
- Giữ `app/streamlit_app.py` làm bản legacy; không sửa `src/`, model artifacts, dataset hoặc reports.

### Đã thêm (Added)
- Demo Streamlit tiếng Việt “Vườn chín” với tải ảnh giới hạn 10 MB, kết quả phân loại, xác suất 9 lớp và giao diện responsive.
- Bảng quản lý nhiệm vụ đa tác tử `context/task-board.md` theo skill `multi-agent-coordinator`.
- Tài liệu kỹ thuật chi tiết trong `docs/`:
  - `docs/installation.md`: Hướng dẫn cài đặt chi tiết trên Windows với `uv` và Python 3.14.
  - `docs/training.md`: Chi tiết pipeline ML, trích xuất 51 đặc trưng, 7 thuật toán và GridSearchCV.
  - `docs/inference.md`: Hướng dẫn sử dụng `FruitClassifier` Python API và ứng dụng Web Streamlit.
  - `docs/results.md`: Báo cáo kết quả thực nghiệm với số liệu thật từ huấn luyện trên 2.507 mẫu.
- Đối chiếu và sửa mô tả HSV, GLCM, đặc trưng hình dạng trong `docs/training.md` để khớp với `src/features.py`.
- Mô hình và artifact đã huấn luyện trong `models/`:
  - `models/classifier.joblib`: Mô hình SVM tối ưu ($C=50$, RBF).
  - `models/scaler.joblib`: Bộ chuẩn hóa `StandardScaler` 51 chiều.
  - `models/config.json`: Toàn bộ siêu tham số và kết quả đánh giá.
- Các biểu đồ thực nghiệm trực quan trong `reports/`:
  - `reports/confusion_matrix.png`: Ma trận nhầm lẫn 9 lớp trên tập test.
  - `reports/roc_curves.png`: Đường cong ROC One-vs-Rest (Macro AUC: 0.9985).
  - `reports/model_comparison.png`: So sánh Accuracy và Macro F1 của 7 thuật toán.
  - `reports/classification_report_SVM.txt`: Bảng chi tiết Precision, Recall, F1 theo từng lớp.

### Đã thay đổi (Changed)
- Thiết kế lại màn hình dự đoán, lịch sử và thông tin mô hình theo chủ đề kiểm định độ chín trái cây; tránh lặp bản ghi lịch sử khi Streamlit rerun.
- `README.md`: Cập nhật toàn diện hướng dẫn cài đặt với `uv`, các tùy chọn huấn luyện, chạy Streamlit và kiểm thử.
- `context/progress.md`: Cập nhật tiến độ hoàn thành 100% tất cả 8 giai đoạn của dự án.
- `context/task-board.md`: Cập nhật tất cả các task đã hoàn thành (`[x] DONE`).

### Đã sửa (Fixed)
- Sửa lỗi `ModuleNotFoundError: No module named 'src'` khi chạy Streamlit từ `app/`; cố định đường dẫn model và báo cáo theo vị trí ứng dụng.
- Sửa lỗi `(label_dir / img_path.stem).with_suffix(".txt")` trong `src/dataset.py`: chuyển thành `label_dir / f"{img_path.stem}.txt"` để nạp chính xác tên file Roboflow có nhiều dấu chấm.
- Sửa lỗi `UnicodeEncodeError: 'charmap'` khi in console tiếng Việt trên Windows trong `src/train.py`.
- Sửa lỗi gõ nhầm thuộc tính `gs.best_params__` trong GridSearchCV (`src/train.py`).
- Sửa lỗi truyền ma trận đặc trưng lệch kích thước `X_val` thay vì `X_test` trong hàm `full_report` (`src/evaluate.py`).

---

## [2026-10-30]

### Đã thêm (Added)
- Thư mục `context/` tuân thủ skill `context-manager`: `README.md`, `project-overview.md`, `architecture.md`, `decisions.md`, `progress.md`, `issues.md`, `changelog.md`, `sessions/2026-10-30.md`.
- `src/dataset.py`: Đọc dataset YOLO, phân tích bounding box, crop đối tượng và resize 128×128.
- `src/features.py`: Trích xuất 51 đặc trưng (Color Histogram, Color Statistics RGB/HSV, LBP Texture, GLCM Texture, Shape metrics từ Otsu mask).
- `src/train.py`: Pipeline huấn luyện 7 mô hình học máy truyền thống, kiểm thử độc lập, GridSearchCV, lưu mô hình joblib và metadata `config.json`.
- `src/evaluate.py`: Bộ công cụ tính metrics và xuất biểu đồ báo cáo (Confusion Matrix, ROC curves, Model Comparison, Feature Importance).
- `src/predict.py`: Bộ nạp mô hình và suy luận thời gian thực `FruitClassifier` phục vụ dự đoán ảnh độc lập hoặc qua Web UI.
- `app/streamlit_app.py`: Giao diện Web phân loại trái cây hiện đại, tùy biến CSS theo skill `frontend-design` (bản sắc nông nghiệp công nghệ cao, màu xanh lá đậm & tươi, nền kem, đầy đủ chức năng dự đoán, lịch sử và thông tin mô hình).
- `tests/test_pipeline.py`: Bộ kiểm thử tự động toàn diện bao gồm dataset parsing, tính toán đặc trưng, đánh giá mô hình và engine dự đoán.

### Đã thay đổi (Changed)
- `requirements.txt`: Thay thế PyTorch/torchvision bằng các thư viện học máy truyền thống và xử lý ảnh (`scikit-learn`, `scikit-image`, `opencv-python-headless`, `scipy`, `seaborn`, `joblib`, `streamlit`).
- Cấu hình môi trường: Tạo virtual environment `.venv` chuẩn Python 3.14.7 thông qua công cụ `uv`.

### Đã sửa (Fixed)
- Phát hiện lỗi làm tròn số thực trong việc chuyển đổi tọa độ YOLO pixel (`int()` gây mất số liệu, sửa bằng `int(round(...))`).
- Xử lý vấn đề môi trường sandbox chặn truy cập Python 3.14 của hệ thống bằng việc tạo venv nội bộ trong thư mục dự án.
