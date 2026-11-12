# Mô tả chương trình phân loại trái cây

## 1. Mục tiêu

Chương trình nhận ảnh trái cây, dự đoán kết hợp **loại quả** và **trạng thái**
thành một trong 9 lớp của model đang nạp. Với cấu hình hiện lưu, các lớp là
Cam, Chuối, Xoài × Chưa chín, Chín, Hỏng. Model là bộ phân loại học máy
truyền thống, không phải mạng CNN.

## 2. Các tính năng

- **Nhận diện ảnh:** kéo-thả hoặc chọn ảnh JPG, PNG, BMP, WEBP; xem trước ảnh,
  gửi dự đoán và xem nhãn, trạng thái, thời gian xử lý cùng độ tin cậy/xác
  suất nếu model hỗ trợ.
- **Giao diện theo kết quả:** sau dự đoán, màu nhấn chuyển theo trạng thái:
  Chưa chín dùng xanh lá, Chín dùng vàng, Hỏng dùng nâu. Khi thả ảnh, vùng
  xem trước có hiệu ứng màu xác nhận thao tác.
- **Tổng quan, lịch sử và thống kê:** xem các lượt phân loại đã ghi, tìm kiếm,
  lọc nhãn, phân trang, xóa lịch sử và xem phân bố theo lớp/trạng thái hoặc
  theo thời gian.
- **Thông tin mô hình:** xem cấu hình, nhãn, chỉ số đánh giá và các biểu đồ
  báo cáo hiện có.
- **API:** kiểm tra trạng thái dịch vụ, nhận ảnh dự đoán, đọc/xóa lịch sử và
  lấy thống kê. Ảnh chỉ được dùng cho suy luận; lịch sử SQLite lưu metadata,
  không lưu nội dung ảnh.

## 3. Công nghệ và vai trò

| Công cụ | Vai trò |
|---|---|
| Python 3.14 | Điều phối tiền xử lý, huấn luyện, suy luận và API. |
| Pillow, NumPy | Đọc/chuyển ảnh RGB; tạo ma trận pixel số làm đầu vào model. |
| OpenCV | Tạo biến thể tăng cường ảnh train như xoay, lật và đổi độ sáng. |
| scikit-learn | `StandardScaler`, PCA, các bộ phân loại, GridSearchCV và metrics. |
| Matplotlib, Seaborn | Sinh learning curve, confusion matrix, ROC và các hình báo cáo. |
| FastAPI, Pydantic, Uvicorn | Khai báo REST API/schema và chạy máy chủ web cục bộ. |
| HTML, CSS, JavaScript | Giao diện nhiều trang, tải ảnh, hiển thị kết quả và biểu đồ. |
| SQLite | Lưu lịch sử dự đoán và cung cấp số liệu cho dashboard/thống kê. |

## 4. Ảnh được chuyển thành model như thế nào?

1. **Nạp ảnh có nhãn:** pipeline đọc các split train, validation và test đã
   chuẩn bị. Ảnh trùng chính xác giữa các split được loại, ưu tiên giữ test,
   rồi validation, rồi train.
2. **Chuyển sang vector:** mỗi ảnh được đổi RGB, resize mặc định về `64 × 64`,
   chuẩn hóa pixel về `[0, 1]` và làm phẳng thành vector `12.288` giá trị.
   Các vector và nhãn được cache trong file `.npz`; `train` kiểm tra cache và
   tiền xử lý lại nếu dữ liệu nguồn thay đổi.
3. **Chuẩn hóa và giảm chiều:** `StandardScaler` chuẩn hóa đầu vào; PCA mặc
   định giữ 95% phương sai. Các bước này được fit trên dữ liệu train, không
   fit trên test.
4. **So sánh và tinh chỉnh:** 9 estimator scikit-learn được so sánh theo
   validation. Ba ứng viên phù hợp nhất được tinh chỉnh bằng 5-fold
   `StratifiedKFold` trên train. Macro F1 dùng để so sánh cân bằng hiệu năng
   giữa các lớp.
5. **Fit mô hình cuối:** dữ liệu train được tăng cường; validation và test
   không được tăng cường. Mô hình cuối được fit lại trên train gốc cộng các
   biến thể, và pipeline cũng huấn luyện model riêng theo từng loại quả.
6. **Đánh giá và lưu:** sau khi chọn xong, kết quả được đo trên test. Model,
   scaler/PCA, nhãn và cấu hình được lưu trong `models/`; bảng số liệu và hình
   được lưu trong `reports/`.

Sơ đồ pipeline và danh sách đầu ra được duy trì trong
[tài liệu huấn luyện](../docs/training.md). Mã tiền xử lý ở
[src/preprocess.py](../src/preprocess.py), huấn luyện ở
[src/train.py](../src/train.py), đánh giá và sinh biểu đồ ở
[src/evaluate.py](../src/evaluate.py).

## 5. Kết quả hiện đang lưu và cách diễn giải

Theo [summary.md](./summary.md), lần chạy được báo cáo chọn **SVM-RBF**:

| Tập | Accuracy | Macro F1 |
|---|---:|---:|
| Validation sau khi chọn/fit mô hình | 90,41% | 0,9030 |
| Test độc lập | 87,04% | 0,8717 |

Model config ghi 1.258 ảnh train, 271 validation, 270 test; sau tăng cường
đã fit trên 2.516 ảnh train gốc và biến thể. Test có 30 ảnh cho mỗi lớp.
Đây là kết quả của artifact/báo cáo đã lưu từ lần huấn luyện trước. Theo
task board, chưa chạy lại huấn luyện sau các bugfix cuối, vì vậy các số liệu
này **chưa xác nhận hiệu năng của phiên bản code cuối**. Cần chạy lại pipeline
trên split hiện hành trước khi dùng làm kết quả thực nghiệm chính thức. Dù đã
xác nhận trên phiên bản được huấn luyện, các chỉ số cũng không cam kết độ
chính xác trên mọi ảnh thực tế.

### Learning curve và dấu hiệu quá khớp

Ở điểm cuối của [learning_curve.csv](./learning_curve.csv), 1.006 mẫu dùng
trong các fold cho train Macro F1 = **1,0000**, còn CV Macro F1 =
**0,8437 ± 0,0217**. Khoảng cách khoảng **0,1563** là dấu hiệu mô hình khớp
rất tốt dữ liệu train nhưng tổng quát hóa kém hơn trên các fold chưa dùng để
fit. Điểm CV tăng từ 0,7982 lên 0,8437 khi số mẫu tăng trong các điểm đo;
đây là xu hướng của lần chạy này, không chứng minh rằng thêm dữ liệu sẽ luôn
cải thiện kết quả.

### Lớp dễ nhầm và lý do kết quả có thể khác nhau

Trong [classification_metrics.csv](./classification_metrics.csv), F1 của
`Mango Spoiled` là **0,646**, thấp hơn các lớp còn lại; `Mango Ripe` đạt
**0,778**. Báo cáo lỗi có các nhầm lẫn `Mango Ripe` → `Mango Spoiled` (7 ảnh)
và `Mango Unripe` → `Mango Spoiled` (4 ảnh). Điều đó góp phần kéo Macro F1
xuống vì Macro F1 tính trung bình đồng đều giữa các lớp.

Đầu vào hiện là pixel RGB thu nhỏ, nên model học các mẫu màu/điểm ảnh có liên
hệ với nhãn trong dữ liệu. Ánh sáng, nền, góc chụp và vùng màu chuyển tiếp
giữa các trạng thái có thể làm ảnh khó phân biệt; đây là các giả thuyết cần
đối chiếu với ảnh trong `misclassified/`, không phải nguyên nhân đã được
chứng minh chỉ từ metrics. Độ tin cậy/xác suất là đầu ra của model hỗ trợ
`predict_proba`, không đồng nghĩa chắc chắn tuyệt đối rằng dự đoán đúng.

## 6. Chạy chương trình

Từ thư mục gốc, `python -m src.train` kiểm tra cache, tiền xử lý nếu cần, huấn
luyện và sinh báo cáo. Chạy giao diện bằng `python -m app.web`, sau đó mở
`http://127.0.0.1:8000`. Chi tiết API và các màn hình được mô tả trong
[docs/inference.md](../docs/inference.md).
