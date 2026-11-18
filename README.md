# 🍊 Hệ Thống Phân Loại Trái Cây & Đánh Giá Chất Lượng Độ Chín bằng Học Sâu (Deep Learning)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CUDA%20%2F%20CPU)-EE4C2C.svg)](https://pytorch.org/)
[![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime%20Web-In--Browser%20Inference-005CED.svg)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Modern%20REST%20API-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Hệ thống ứng dụng mạng nơ-ron tích chập (**Convolutional Neural Networks - CNN**) và học chuyển giao (**Transfer Learning**) hiện đại để tự động trích chọn đặc trưng không gian đa tỷ lệ từ ảnh chụp trái cây. Dự án giải quyết trọn vẹn và đồng thời hai bài toán thực tiễn quan trọng:

1. **Phân loại Chủng loại & Độ chín (9 phân lớp)**: Nhận diện chính xác 3 loại trái cây phổ biến (*Cam, Chuối, Xoài*) qua 3 giai đoạn sinh trưởng (*Chưa chín / Xanh*, *Chín*, *Bị hỏng / Thối*).
2. **Đánh giá Phẩm cấp Thương phẩm & An toàn Thực phẩm**: Tự động xếp hạng thương phẩm thành 3 cấp độ (*Đạt chuẩn Loại 1*, *Lưu kho / Chế biến Loại 2*, *Kém chất lượng / Loại bỏ*) kèm khuyến nghị bảo quản và cảnh báo vi sinh vật, nấm mốc.

> 💡 **Ghi chú chuyển đổi kiến trúc:** Hệ thống hiện tại sử dụng toàn bộ pipeline Học Sâu thế hệ mới trong thư mục [`src_dl/`](src_dl/) và kiến trúc Edge Inference trên trình duyệt [`frontend/`](frontend/). Tài liệu và mã nguồn học máy truyền thống cũ (Handcrafted Features / Raw Pixels + scikit-learn) đã được lưu trữ tham khảo tại [`README_ml.md`](README_ml.md) và thư mục [`src/`](src/).

---

## 📑 Mục Lục

1. [Điểm Nổi Bật Của Hệ Thống](#-điểm-nổi-bật-của-hệ-thống)
2. [Kiến Trúc Kỹ Thuật (Pipeline Deep Learning)](#-kiến-trúc-kỹ-thuật-pipeline-deep-learning)
3. [Tập Dữ Liệu Thực Nghiệm (Dataset)](#-tập-dữ-liệu-thực-nghiệm-dataset)
4. [Cấu Trúc Thư Mục Dự Án](#-cấu-trúc-thư-mục-dự-án)
5. [Cài Đặt & Môi Trường](#-cài-đặt--môi-trường)
6. [Huấn Luyện & Tối Ưu Mô Hình (`src_dl/`)](#-huấn-luyện--tối-ưu-mô-hình-src_dl)
7. [Xuất Khẩu ONNX & Suy Luận Trên Trình Duyệt](#-xuất-khẩu-onnx--suy-luận-trên-trình-duyệt)
8. [Khởi Chạy Ứng Dụng Web & API](#-khởi-chạy-ứng-dụng-web--api)
9. [Báo Cáo & Kết Quả Thực Nghiệm](#-báo-cáo--kết-quả-thực-nghiệm)
10. [Triển Khai Sản Xuất (Vercel Serverless)](#-triển-khai-sản-xuất-vercel-serverless)

---

## 🌟 Điểm Nổi Bật Của Hệ Thống

* **Trích xuất đặc trưng sâu tự động (Deep Feature Extraction)**: Thay thế hoàn toàn 90 đặc trưng thủ công và vector pixel thô bằng CNN Backbone tiền huấn luyện trên tập ImageNet (**MobileNetV3-Small**, **MobileNetV2**, **ResNet-18**), nắm bắt đặc trưng ngữ nghĩa và hoa văn vỏ quả vượt trội.
* **Suy luận thời gian thực trực tiếp trên Client (Edge AI via ONNX Runtime Web)**: Mô hình học sâu được nén và đóng gói thành tệp ONNX siêu nhẹ (~4.2 MB). Trình duyệt của người dùng có thể tự thực hiện suy luận cục bộ (Client-side Inference) thông qua WebAssembly/WebGL mà không cần gửi ảnh lên server, đạt độ trễ gần như tức thì và chi phí vận hành máy chủ bằng 0.
* **Giao diện Web SPA Hiện Đại & Thân Thiện**:
  - Hỗ trợ phím tắt **`Ctrl + V`** để dán ảnh chụp màn hình hoặc dán trực tiếp từ bộ nhớ tạm (Clipboard).
  - Kéo & thả ảnh (Drag-and-Drop), duyệt tệp từ máy hoặc chụp từ Camera.
  - **Dải ảnh mẫu thử nghiệm 1-chạm**: Xếp gọn gàng theo 1 hàng ngang cuộn mượt mà, giúp kiểm thử nhanh chóng các trạng thái quả.
  - **Thanh điều hướng thu gọn (Collapsible Sidebar)**: Tối ưu không gian hiển thị trên cả máy tính bảng và màn hình lớn.
  - **Cơ chế tự động dọn Cache**: Tự động phát hiện phiên bản ứng dụng mới (`version.json`) để xóa cache trình duyệt, đảm bảo giao diện luôn được cập nhật.
* **Đánh giá đa chiều**: Cung cấp xác suất phân bố trực quan trên 9 lớp, chỉ số tự tin (Confidence score), cấp độ thương phẩm và lời khuyên sử dụng/bảo quản cụ thể theo từng loại quả.

---

## 🔬 Kiến Trúc Kỹ Thuật (Pipeline Deep Learning)

Hệ thống hoạt động theo quy trình khép kín từ tiền xử lý, tăng cường dữ liệu, trích chọn đặc trưng đến phân loại và xuất khẩu:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            PIPELINE HỌC SÂU (src_dl/)                            │
│                                                                                  │
│  Ảnh gốc (RGB)                                                                   │
│       │                                                                          │
│       ▼  [Train time] Tăng cường ảnh (Data Augmentation)                         │
│          • Lật ngang ngẫu nhiên (Random Horizontal Flip, p = 0.5)                 │
│          • Xoay góc ngẫu nhiên (Random Rotation, ±15°)                           │
│          • Biến thiên màu sắc (ColorJitter: Độ sáng, tương phản ±15%)            │
│       │                                                                          │
│       ▼  Resize chuẩn: 224 × 224 pixel                                           │
│       │                                                                          │
│       ▼  Chuẩn hóa ImageNet: Mean [0.485, 0.456, 0.406], Std [0.229, 0.224, 0.225]│
│       │                                                                          │
│       ▼  CNN Backbone Feature Extractor (Frozen Pretrained Weights)              │
│          • MobileNetV3-Small (576-dim)  [Mặc định - Tối ưu suy luận web siêu nhẹ]│
│          • MobileNetV2       (1280-dim) [Cân bằng độ chính xác và tốc độ]        │
│          • ResNet-18         (512-dim)  [Độ chính xác cao]                       │
│       │                                                                          │
│       ▼  Adaptive Average Pooling 2D + Làm phẳng vector đặc trưng                │
│       │                                                                          │
│       ▼  Chuẩn hóa StandardScaler + Giảm chiều PCA (giữ 95% phương sai)          │
│       │                                                                          │
│       ▼  PyTorch MLP Classifier Head                                             │
│          • Linear(in_features, 128) -> BatchNorm1d -> ReLU -> Dropout(p=0.3)     │
│          • Linear(128, 9) -> Output Logits                                       │
│       │                                                                          │
│       ├──────────────────────────────────────┬───────────────────────────────────┤
│       ▼                                      ▼                                   │
│  [Huấn luyện & Báo cáo]                 [Triển khai Suy luận Client/Server]     │
│  • Huấn luyện GPU CUDA (RTX 3050) / CPU • Đóng gói end-to-end: export_onnx.py    │
│  • Tự động sinh 6 biểu đồ & chỉ số      • Chạy cục bộ bằng ONNX Runtime Web     │
│  • Lưu checkpoint: models_dl/           • Hoặc REST API backend: app/web/        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Thông Số Kỹ Thuật Các Backbone Hỗ Trợ:

| Backbone | Số tham số | Kích thước Embedding | Kích thước ONNX xuất khẩu | Ưu điểm & Ứng dụng |
|---|:---:|:---:|:---:|---|
| **MobileNetV3-Small** *(Mặc định)* | ~2.5M | 576 chiều | **~4.4 MB** | Siêu nhẹ, độ trễ mili-giây, hoàn hảo cho Edge AI Web. |
| **MobileNetV2** | ~3.5M | 1.280 chiều | **~11.0 MB** | Khả năng phân tách đặc trưng không gian rất tốt. |
| **ResNet-18** | ~11.7M | 512 chiều | **~45.4 MB** | Khả năng tổng quát hóa cao, biểu diễn đặc trưng mạnh mẽ. |

---

## 📊 Tập Dữ Liệu Thực Nghiệm (Dataset)

Bộ dữ liệu phân loại được lưu trữ và tổ chức khoa học tại thư mục [`data/processed/dataset/`](data/README.md), gồm **9 lớp** được kết hợp từ 3 loại trái cây và 3 cấp độ chín/chất lượng:

* **Cam**: `cam-chua-chin` (Cam xanh), `cam-chin` (Cam chín vàng/cam), `cam-hong` (Cam bị mốc/thối/hỏng).
* **Chuối**: `chuoi-chua-chin` (Chuối xanh), `chuoi-chin` (Chuối chín vàng), `chuoi-hong` (Chuối thâm đen/lên men/hỏng).
* **Xoài**: `xoai-chua-chin` (Xoài xanh), `xoai-chin` (Xoài chín vàng), `xoai-hong` (Xoài đốm đen/thối/hỏng).

### Bảng Phân Bố Số Lượng Ảnh Đã Xác Thực Thực Tế:

| Nhãn Lớp (Class Label) | Tập Huấn Luyện (Train) | Tập Xác Thực (Validation) | Tập Kiểm Thử (Test) | Tổng Cộng |
|---|:---:|:---:|:---:|:---:|
| `cam-chin` | 140 | 30 | 30 | **200** |
| `cam-chua-chin` | 140 | 30 | 30 | **200** |
| `cam-hong` | 140 | 30 | 30 | **200** |
| `chuoi-chin` | 140 | 30 | 30 | **200** |
| `chuoi-chua-chin` | 140 | 30 | 30 | **200** |
| `chuoi-hong` | 140 | 30 | 30 | **200** |
| `xoai-chin` | 140 | 30 | 30 | **200** |
| `xoai-chua-chin` | 140 | 30 | 30 | **200** |
| `xoai-hong` | 140 | 30 | 30 | **200** |
| **Tổng số ảnh** | **1.260** *(70.0%)* | **270** *(15.0%)* | **270** *(15.0%)* | **1.800 ảnh** |

> ✅ **Đảm bảo tính khoa học:** 100% tệp ảnh đều giải mã hợp lệ (0 corrupt). Tỷ lệ phân hoạch chuẩn $70\% - 15\% - 15\%$ cân bằng tuyệt đối giữa các lớp, không xảy ra hiện tượng mất cân bằng dữ liệu (class imbalance).

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
fruit-classification/
├── app/web/                    # Ứng dụng Web FastAPI Backend & REST API
│   ├── config.py               # Cấu hình đường dẫn, model DL và static files
│   ├── routes/                 # Các REST API: prediction, health, model_info, history
│   └── services/               # Prediction service nạp mô hình DL & SQLite history
├── frontend/                   # Ứng dụng Web Frontend SPA (Vanilla JS + ONNX Web)
│   ├── index.html              # Shell chính với Navbar thu gọn, layout hiện đại
│   ├── version.json            # Quản lý phiên bản để auto clear client cache
│   ├── pages/                  # Các màn hình: dashboard, recognition, model-info, history...
│   ├── static/                 # CSS theme (dark/light, ripeness palettes) & Core JS
│   │   ├── css/style.css       # Toàn bộ giao diện tùy biến, responsive
│   │   └── js/                 # Router, Model Manager, ONNX Runner, UI controllers
│   └── models/                 # Các mô hình ONNX phục vụ In-Browser Inference
├── data/
│   ├── README.md               # Đặc tả chi tiết tập dữ liệu và báo cáo kiểm toán audit
│   └── processed/dataset/      # 1.800 ảnh train / val / test phân bổ 9 lớp
├── models_dl/                  # Trọng số mô hình đã huấn luyện (classifier_dl.joblib, config.json)
├── reports_dl/                 # Báo cáo thực nghiệm: metrics, confusion matrix, ROC curves, loss
├── src_dl/                     # Mã nguồn cốt lõi nhánh Học Sâu (Deep Learning)
│   ├── backbone.py             # Quản lý các CNN Backbone (MobileNetV3, MobileNetV2, ResNet18)
│   ├── preprocess_dl.py        # Tiền xử lý, chuẩn hóa ImageNet, Data Augmentation
│   ├── torch_classifier.py     # Kiến trúc PyTorch MLP Head & bộ theo dõi LossTracker
│   ├── benchmark_reporter.py   # Đo đạc FLOPs, số tham số, độ trễ và xuất báo cáo học thuật
│   ├── train_dl.py             # Quy trình huấn luyện tự động với GPU CUDA / CPU
│   └── predict_dl.py           # Bộ suy luận độc lập FruitClassifierDL
├── export_onnx.py              # Script xuất toàn bộ pipeline sang định dạng ONNX
├── README_ml.md                # Tài liệu lưu trữ nhánh học máy truyền thống cũ (src/)
├── requirements.txt            # Danh mục thư viện Python sản xuất
└── vercel.json                 # Cấu hình triển khai Serverless Vercel
```

---

## 🛠️ Cài Đặt & Môi Trường

### 1. Khởi tạo môi trường ảo Python (khuyến nghị Python 3.10 - 3.14):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Cài đặt các thư viện sản xuất:

```powershell
pip install -r requirements.txt
```

### 3. Cài đặt PyTorch hỗ trợ GPU CUDA (Tùy chọn nếu máy có card rời NVIDIA):

Nếu bạn có GPU rời NVIDIA (ví dụ: RTX 3050 / RTX 4060) và muốn tăng tốc độ huấn luyện:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
*(Nếu chạy trên CPU, hệ thống vẫn hoạt động hoàn toàn mượt mà và tự động nhận diện thiết bị tính toán)*.

---

## 🚀 Huấn Luyện & Tối Ưu Mô Hình (`src_dl/`)

Quy trình huấn luyện trong `src_dl/train_dl.py` tự động tích hợp: trích xuất đặc trưng sâu, chuẩn hóa, huấn luyện MLP Head và xuất đầy đủ biểu đồ đánh giá.

### Huấn luyện mô hình mặc định (MobileNetV3-Small trên GPU CUDA/CPU):
```powershell
python -m src_dl.train_dl --backbone mobilenetv3_small --epochs 15 --device auto
```

### Huấn luyện và so sánh cùng lúc cả 3 kiến trúc:
```powershell
python -m src_dl.train_dl --backbone all --epochs 15 --device auto
```

### Huấn luyện nối tiếp từ checkpoint đã lưu (`--resume`):
```powershell
python -m src_dl.train_dl --resume --epochs 10
```

### Các cờ tùy chọn nâng cao:
* `--device {auto,cuda,cpu}`: Chỉ định phần cứng huấn luyện.
* `--lr 0.001`: Tùy chỉnh tốc độ học (Learning Rate).
* `--no-pca`: Tắt tầng giảm chiều PCA nếu muốn giữ nguyên toàn bộ số chiều đặc trưng CNN.
* `--force-preprocess`: Buộc trích xuất lại đặc trưng ảnh dù tệp cache `.npz` đã tồn tại.

---

## 🌐 Xuất Khẩu ONNX & Suy Luận Trên Trình Duyệt

Để đưa mô hình học sâu vào hoạt động trực tiếp trong trình duyệt người dùng qua **ONNX Runtime Web**, chạy lệnh xuất khẩu:

```powershell
python export_onnx.py --backbone all
```

Lệnh này sẽ tự động ghép nối toàn bộ đồ thị tính toán:
$$\text{Ảnh } (224 \times 224 \times 3) \xrightarrow{\text{Backbone}} \text{CNN Features} \xrightarrow{\text{Scaler + PCA}} \text{MLP Layers} \xrightarrow{\text{Softmax}} \text{9 Xác suất Logits}$$

Các tệp mô hình ONNX tối ưu sẽ được lưu trực tiếp vào thư mục `frontend/models/`:
* `model_mobilenetv3_small.onnx` (~4.4 MB) — mô hình mặc định cho web.
* `model_mobilenetv2.onnx` (~11.0 MB).
* `model_resnet18.onnx` (~45.4 MB).
* `model.onnx` — liên kết trỏ đến mô hình có độ chính xác cao nhất.

---

## 💻 Khởi Chạy Ứng Dụng Web & API

Khởi động máy chủ phát triển cục bộ bằng FastAPI:

```powershell
python -m app.web
```

* 🌍 **Giao diện Web Ứng Dụng**: Mở trình duyệt truy cập: **`http://127.0.0.1:8000`**
* 📖 **Tài liệu API Swagger tương tác**: **`http://127.0.0.1:8000/docs`**

### Các Thao Tác Trải Nghiệm Nhanh Trên Web:
1. **Dán ảnh siêu tốc (`Ctrl + V`)**: Bấm tổ hợp phím `Ctrl + V` bất kỳ lúc nào trên trang để dán ngay ảnh vừa chụp màn hình hoặc copy từ web.
2. **Thử nghiệm 1-chạm**: Nhấp vào bất kỳ ảnh mẫu nào trên thanh cuộn ngang để xem kết quả nhận diện ngay lập tức.
3. **Chuyển đổi Engine suy luận**: Tại mục Cài đặt/Mô hình, bạn có thể linh hoạt chuyển đổi giữa suy luận cục bộ bằng **ONNX Runtime Web** (In-Browser) hoặc gửi qua **REST API Server**.
4. **Theo dõi Lịch sử**: Toàn bộ kết quả phân loại, thời gian thực thi và hình ảnh đều được lưu trữ trong trang Lịch sử.

---

## 📈 Báo Cáo & Kết Quả Thực Nghiệm

Sau mỗi phiên huấn luyện, hệ thống tự động xuất 6 tài liệu học thuật theo chuẩn nghiên cứu vào thư mục [`reports_dl/`](reports_dl/):

1. **`metrics.json`**: Thống kê chi tiết điểm số `Accuracy`, `Macro F1`, `Weighted F1`, `Precision`, `Recall` trên từng tập dữ liệu.
2. **`confusion_matrix.png`**: Ma trận nhầm lẫn kích thước $9 \times 9$ biểu diễn trực quan độ chính xác giữa nhãn thực tế và nhãn dự đoán.
3. **`roc_curve.png`**: Đường cong đặc trưng hoạt động của bộ thu (Receiver Operating Characteristic) đa lớp kèm chỉ số diện tích dưới đường cong (Macro AUC).
4. **`training_history.png` / `loss_curve.png`**: Đồ thị diễn tiến hàm mất mát (Loss) và độ chính xác (Accuracy) qua từng Epoch, giúp giám sát hiện tượng Overfitting/Underfitting.
5. **`model_comparison.png`**: Biểu đồ so sánh trực quan đa tiêu chí giữa MobileNetV3-Small, MobileNetV2 và ResNet-18.
6. **`benchmark.csv`**: Bảng dữ liệu định lượng về độ phức tạp tính toán (FLOPs, số lượng tham số, kích thước file ONNX, độ trễ suy luận mili-giây trên CPU/GPU).

---

## ☁️ Triển Khai Sản Xuất (Vercel Serverless)

Dự án đã được cấu hình tối ưu để sẵn sàng triển khai lên nền tảng **Vercel**:

* Tệp cấu hình [`vercel.json`](vercel.json) thiết lập rewrite thông minh chuyển hướng mọi yêu cầu frontend SPA và ánh xạ `/api/(.*)` sang Serverless Function [`api/index.py`](api/index.py).
* Bộ thư viện sản xuất [`requirements.txt`](requirements.txt) được tinh giản (<200 MB), hoàn toàn tương thích với giới hạn dung lượng nghiêm ngặt của AWS Lambda và Vercel.
* Do hỗ trợ **ONNX Runtime Web**, ứng dụng thậm chí có thể hoạt động hoàn toàn như một trang Web tĩnh (Static Web App), người dùng ở khắp nơi trên thế giới đều có thể phân loại ảnh với tốc độ tức thì mà không cần server GPU đắt đỏ.

```powershell
# Triển khai nhanh bằng Vercel CLI
vercel --prod
```

---

## 📜 Giấy Phép & Tác Quyền

Dự án được phát triển phục vụ mục đích nghiên cứu, học tập và ứng dụng thực tiễn trong phân loại nông sản. Mã nguồn được phát hành theo giấy phép [MIT License](LICENSE). Mọi đóng góp, phản hồi và câu hỏi vui lòng mở Issue hoặc Pull Request trên repository!
