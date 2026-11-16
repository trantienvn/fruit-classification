# Hệ Thống Phân Loại Độ Chín & Chất Lượng Trái Cây bằng Học Sâu (Deep Learning)

Dự án áp dụng mạng nơ-ron tích chập (Convolutional Neural Network - CNN) và học chuyển giao (Transfer Learning) để tự động trích chọn đặc trưng không gian từ ảnh chụp, giải quyết đồng thời hai bài toán quan trọng:
1. **Phân loại chủng loại & Độ chín của trái cây** (Cam, Chuối, Xoài × Chín mọng / Chưa chín / Bị hỏng).
2. **Đánh giá chất lượng thương phẩm** (Đạt chuẩn Loại 1, Đạt chuẩn lưu kho Loại 2, hoặc Kém chất lượng / Loại bỏ) kèm khuyến nghị an toàn thực phẩm.

---

## Mục Lục Cơ Sở Lý Thuyết & Phương Pháp

Dự án được xây dựng và tổ chức bám sát đề cương nghiên cứu:

- **Chương 1: Cơ sở lý thuyết**
  - Mạng nơ-ron tích chập (CNN), Học chuyển giao (Transfer Learning) và Fine-tuning.
  - Cơ sở sinh hóa về độ chín và các tiêu chí đánh giá chất lượng trái cây.
  - Các chỉ số đánh giá mô hình phân loại: Accuracy, Precision, Recall, Macro F1, Confusion Matrix, ROC-AUC.
- **Chương 2: Mô hình / Phương pháp đề xuất**
  - Sơ đồ quy trình từ ảnh RGB (224×224) $\rightarrow$ CNN Backbone (MobileNetV3-Small) $\rightarrow$ Trích xuất đặc trưng sâu tự động (576 chiều) $\rightarrow$ Tầng phân loại PyTorch MLP (CUDA/GPU).
  - Tiền xử lý chuẩn hóa ImageNet (Mean/Std), tăng cường dữ liệu (Data Augmentation: Random Flip, Rotation, ColorJitter).
  - Chiến lược xử lý song hành 2 thuộc tính: Độ chín (Ripeness) và Chất lượng (Quality).
- **Chương 3: Kết quả thực nghiệm & Giao diện ứng dụng**
  - Thống kê dữ liệu, chia tập Train / Val / Test độc lập.
  - Sơ đồ xu hướng Loss và Accuracy qua từng Epoch (`loss_curve.png`, `epoch_training.png`), kiểm soát Overfitting/Underfitting.
  - Giao diện Web hiện đại: kéo thả ảnh, tải tệp và **hỗ trợ phím tắt `Ctrl + V` để dán ảnh trực tiếp từ clipboard**.

---

## Cấu Trúc Dự Án

```
fruit-classification/
├── api/                    # Vercel Serverless Entrypoint (index.py)
├── app/web/                # Ứng dụng Web FastAPI & Giao diện
│   ├── frontend/           # HTML, CSS hiện đại, Vanilla JS modules
│   │   ├── pages/          # dashboard, recognition, model-info, history, statistics
│   │   └── static/         # CSS và JavaScript xử lý drag-drop & Ctrl+V clipboard
│   ├── routes/             # REST API: prediction, health, model_info, history...
│   └── services/           # History SQLite, Prediction service
├── data/                   # Bộ dữ liệu ảnh raw & processed
├── models_dl/              # Trọng số mô hình học sâu (classifier_dl.joblib, config.json)
├── reports_dl/             # Sơ đồ Loss curve, tiến trình Epoch, báo cáo thực nghiệm
├── src_dl/                 # Mã nguồn Học Sâu (Deep Learning)
│   ├── backbone.py         # CNN Backbone (MobileNetV3-Small ImageNet)
│   ├── preprocess_dl.py    # Chuẩn hóa ImageNet, Data Augmentation, Dataset & DataLoader
│   ├── torch_classifier.py # Mô hình PyTorch MLP Head, loss tracker, xuất biểu đồ Loss
│   ├── train_dl.py         # Quy trình huấn luyện & tiếp tục huấn luyện (--resume)
│   └── predict_dl.py       # Bộ suy luận FruitClassifierDL độc lập
├── tests/                  # Kiểm thử tự động (pytest)
├── requirements.txt        # Phụ thuộc sản xuất tối ưu dung lượng (<250MB cho Vercel)
└── vercel.json             # Cấu hình rewrite và deploy Vercel
```

---

## Cài Đặt Môi Trường

Khuyến nghị sử dụng Python 3.12 - 3.14 trên Windows hoặc Linux:

```powershell
# Tạo môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài đặt các thư viện sản xuất
pip install -r requirements.txt
```

Nếu máy có card đồ họa rời NVIDIA (ví dụ RTX 3050 Laptop GPU) và muốn tận dụng CUDA:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Huấn Luyện Mô Hình Học Sâu (Deep Learning)

### 1. Huấn luyện từ đầu (Single-pass trên GPU CUDA):
```powershell
python -m src_dl.train_dl --models torch --epochs 15 --device cuda
```
*(Hệ thống tự động trích xuất đặc trưng sâu 576 chiều qua CNN, huấn luyện tầng MLP và xuất trực tiếp biểu đồ `loss_curve.png` và `epoch_training.png` sang `reports_dl/`)*.

### 2. Tiếp tục huấn luyện với số Epoch tùy chỉnh (`--resume`):
```powershell
python -m src_dl.train_dl --resume --epochs 10 --device cuda
```
*(Trọng số mô hình được nạp lại và tiếp tục học thêm 10 Epochs, đường cong Loss nối tiếp mượt mà từ chu kỳ trước đó)*.

---

## Chạy Ứng Dụng Web

Khởi chạy ứng dụng Web cục bộ bằng FastAPI:

```powershell
python -m app.web
```

- Mở trình duyệt tại: **http://127.0.0.1:8000**
- Tài liệu tương tác API Swagger Docs: **http://127.0.0.1:8000/docs**

### Điểm nổi bật trên giao diện:
- **Dán ảnh cực nhanh từ bộ nhớ tạm**: Chỉ cần nhấn `Ctrl + V` bất kỳ lúc nào trên màn hình để dán ảnh chụp màn hình hoặc ảnh copy từ internet.
- **Kéo thả & Xem trước trực quan**: Hỗ trợ kéo thả ảnh (Drag & Drop) mượt mà.
- **Đánh giá 3 chiều**: Chủng loại quả · Độ chín · Tiêu chuẩn chất lượng thương phẩm.
- **Lời khuyên tiêu dùng**: Đưa ra hướng dẫn cụ thể về bảo quản, thời gian sử dụng, cách ủ chín tự nhiên hoặc cảnh báo nấm mốc.
- **Phân bố xác suất**: Biểu đồ thanh ngang xác suất trên toàn bộ 9 lớp nhận diện.

---

## Triển Khai Lên Vercel

Dự án đã được cấu hình tối ưu để chạy mượt mà trên **Vercel Serverless Functions**:
- `requirements.txt` siêu nhẹ (< 200MB, vượt xa ngưỡng giới hạn 500MB của AWS Lambda).
- SQLite database tự động lưu vào `/tmp/history.sqlite3`.
- Entry point `api/index.py` tích hợp middleware xử lý định tuyến tự động.

Triển khai bằng Git:
```powershell
git add .
git commit -m "feat: cap nhat hoc sau CNN, ho tro Ctrl+V clipboard va giao dien hien dai"
git push origin main
```
