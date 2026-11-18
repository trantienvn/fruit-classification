# 🍊 Phân Loại Trái Cây — Nhánh Deep Learning (`src_dl/`)

> **Tài liệu này mô tả nhánh học sâu** (`src_dl/`) sử dụng CNN để trích xuất đặc trưng tự động,
> kết hợp với các bộ phân loại scikit-learn quen thuộc.
> Nhánh học máy truyền thống vẫn hoàn toàn nguyên vẹn trong `src/`.

---

## Mục Lục

1. [Tổng Quan & So Sánh](#1-tổng-quan--so-sánh)
2. [Kiến Trúc Pipeline](#2-kiến-trúc-pipeline)
3. [Yêu Cầu Cài Đặt](#3-yêu-cầu-cài-đặt)
4. [Hướng Dẫn Chạy](#4-hướng-dẫn-chạy)
5. [Cấu Trúc Thư Mục](#5-cấu-trúc-thư-mục)
6. [Tham Số & Tuỳ Chỉnh](#6-tham-số--tuỳ-chỉnh)
7. [Lý Thuyết: CNN Feature Extraction](#7-lý-thuyết-cnn-feature-extraction)
8. [Đánh Giá Rủi Ro](#8-đánh-giá-rủi-ro)
9. [Lưu Ý Quan Trọng](#9-lưu-ý-quan-trọng)

---

## 1. Tổng Quan & So Sánh

| Tiêu chí | `src/` (ML truyền thống) | `src_dl/` (CNN + ML) |
|---|---|---|
| **Trích xuất đặc trưng** | Thủ công (pixel phẳng hoặc 90 đặc trưng màu/kết cấu/hình dạng) | Tự động qua CNN backbone (MobileNetV3-Small) |
| **Đặc trưng** | Pixel RGB 64×64 → 12,288 chiều | CNN embedding → 576 chiều |
| **Backbone** | Không có | MobileNetV3-Small pretrained (ImageNet), **frozen** |
| **Classifier** | 9 bộ sklearn (LR, SVM, KNN, RF, GB, DT, LDA, QDA, GNB) | 9 bộ sklearn giống hệt `src/` |
| **Tăng cường** | Trên pixel vectors (sau khi cache) | Trên ảnh gốc (trước khi trích xuất CNN) |
| **Yêu cầu GPU** | Không | Không (CPU-first) |
| **Phụ thuộc bổ sung** | Không | `torch`, `torchvision` |
| **Artifact mô hình** | `models/` | `models_dl/` |
| **Cache đặc trưng** | `data/processed/numeric/` | `data/processed/dl_features/` |

### Lý Do Chọn Kiến Trúc Này

Thay vì huấn luyện CNN end-to-end (đòi hỏi dữ liệu lớn, GPU, thời gian lâu), nhánh `src_dl/` áp dụng **Transfer Learning ở mức trích xuất đặc trưng**:

1. **Frozen backbone**: Giữ toàn bộ kiến thức từ ImageNet, không cần fine-tune.
2. **Lightweight head**: Classifier sklearn — dễ giải thích, nhanh huấn luyện, không cần GPU.
3. **So sánh trực tiếp**: Dùng cùng 9 bộ phân loại với `src/` → kết quả so sánh công bằng.

---

## 2. Kiến Trúc Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE src_dl/                              │
│                                                                  │
│  Ảnh gốc (H×W×3 uint8)                                          │
│       │                                                          │
│       ▼  [Chỉ train] Tăng cường (xoay ±15°, lật 50%, sáng ±15%)│
│       │                                                          │
│       ▼  Resize → 224×224                                        │
│       │                                                          │
│       ▼  Chuẩn hoá ImageNet (mean/std)                           │
│       │                                                          │
│       ▼  MobileNetV3-Small (frozen)                              │
│          └─ features (conv blocks)                               │
│          └─ Adaptive Avg Pool                                    │
│       │                                                          │
│       ▼  Embedding vector (576 chiều, float32)                   │
│       │                                                          │
│       ├──────────────────────────────────────────────────────┐   │
│       ▼  Lưu cache: data/processed/dl_features/*.npz         │   │
│                                                              │   │
│  [Train time]                                               [Inference]
│       │                                                              │
│       ▼  StandardScaler                                             │
│       ▼  PCA (giữ 95% phương sai) [tuỳ chọn]                       │
│       ▼  GridSearchCV (StratifiedKFold-5)                           │
│       ▼  9 bộ phân loại sklearn                                     │
│       │                                                             │
│       ▼  Lưu: models_dl/classifier_dl.joblib + config.json         │
│              reports_dl/summary_dl.txt                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Backbone: MobileNetV3-Small

| Thông số | Giá trị |
|---|---|
| Tên | `mobilenetv3_small` |
| Nguồn | `torchvision.models.mobilenet_v3_small` |
| Trọng số | ImageNet1K_V1 |
| Số tham số | ~2.5 triệu |
| Đầu vào | 224 × 224 × 3 (RGB) |
| Đầu ra embedding | 576 chiều (sau Adaptive Avg Pool) |
| Frozen | Có — không cập nhật gradient |
| Device | CPU (không cần GPU) |

---

## 3. Yêu Cầu Cài Đặt

### 3.1. Yêu Cầu Hệ Thống

- Python 3.14 (hoặc 3.11+)
- RAM: tối thiểu 4 GB (8 GB khuyến nghị)
- Ổ đĩa: ~500 MB cho trọng số PyTorch + torchvision

### 3.2. Cài Đặt Phụ Thuộc Cơ Bản

Trước tiên, cài đặt `requirements.txt` nếu chưa có:

```bash
pip install -r requirements.txt
```

### 3.3. Cài Đặt Phụ Thuộc DL

```bash
pip install -r requirements_dl.txt
```

Hoặc cài trực tiếp:

```bash
pip install torch torchvision
```

Dùng `uv` (nhanh hơn):

```bash
uv pip install torch torchvision
```

> **CPU-only build** (nhẹ hơn, không cần CUDA):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### 3.4. Kiểm Tra Cài Đặt

```bash
python -m src_dl.backbone
```

Kết quả mong đợi:
```
Backbone: mobilenetv3_small
Embedding dim: 576
Embedding shape: (576,), dtype: float32
OK — backbone hoạt động bình thường.
```

---

## 4. Hướng Dẫn Chạy

### Bước 1 — Trích Xuất CNN Embedding (Tiền Xử Lý)

```bash
python -m src_dl.preprocess_dl
```

Lệnh này sẽ:
1. Đọc ảnh từ `data/processed/dataset/` (train/val/test)
2. Áp dụng tăng cường dữ liệu trên ảnh gốc (chỉ tập train)
3. Tải trọng số MobileNetV3-Small từ torchvision (lần đầu có thể mất ~1 phút để tải)
4. Trích xuất embedding 576 chiều cho mỗi ảnh
5. Lưu cache tại `data/processed/dl_features/mobilenetv3_small/dl_dataset.npz`

**Tuỳ chọn:**
```bash
python -m src_dl.preprocess_dl \
    --data-dir data/processed/dataset \
    --cache-dir data/processed/dl_features/mobilenetv3_small \
    --batch-size 64 \
    --force          # trích xuất lại dù cache đã tồn tại
```

> ⏱️ **Thời gian ước tính (CPU):** ~5–15 phút cho 1.260 ảnh train (kèm augmentation × 2 bản tăng cường = 3.780 ảnh), trên tổng tập dữ liệu gốc 1.800 ảnh

---

### Bước 2 — Huấn Luyện Mô Hình

```bash
python -m src_dl.train_dl
```

Lệnh này sẽ:
1. Kiểm tra/tạo cache CNN embedding (tự động chạy Bước 1 nếu chưa có)
2. Tải embedding từ `.npz`
3. Huấn luyện 9 bộ phân loại với GridSearchCV (5-fold StratifiedKFold)
4. Chọn top-3 mô hình tốt nhất theo F1-macro trên tập val
5. Lưu mô hình tốt nhất: `models_dl/classifier_dl.joblib`
6. Lưu báo cáo: `reports_dl/summary_dl.txt`

**Tuỳ chọn:**
```bash
python -m src_dl.train_dl \
    --no-pca              # tắt PCA (không giảm chiều)
    --no-gridsearch       # bỏ qua GridSearch (nhanh hơn, kết quả kém hơn)
    --top-k 5             # giữ top-5 thay vì top-3
    --force-preprocess    # trích xuất lại embedding trước khi train
    --verbose             # logging chi tiết
```

> ⏱️ **Thời gian ước tính:** 10–60 phút (phụ thuộc vào GridSearch và số CPU)

---

### Bước 3 — Dự Đoán

```bash
# Dự đoán một ảnh
python -m src_dl.predict_dl <đường_dẫn_ảnh>

# Ví dụ:
python -m src_dl.predict_dl data/raw/mango_ripe.jpg
```

Kết quả mẫu:
```
Kết quả dự đoán cho: data/raw/mango_ripe.jpg
  Lớp:        Mango_Ripe
  Độ tin cậy: 94.32%
  Backbone:   mobilenetv3_small

Xác suất tất cả lớp:
  Mango_Ripe                     0.9432  ██████████████████████████████
  Mango_Unripe                   0.0401  █
  Orange_Ripe                    0.0098
  ...
```

---

### Dùng Như Thư Viện Python

```python
from src_dl.predict_dl import FruitClassifierDL

# Khởi tạo và tải mô hình
clf = FruitClassifierDL(model_dir="models_dl")
clf.load()

# Dự đoán từ file
result = clf.predict_from_path("ảnh_trái_cây.jpg")
print(result["label"])        # "Mango_Ripe"
print(result["confidence"])   # 0.9432

# Dự đoán từ PIL Image
from PIL import Image
img = Image.open("ảnh.jpg")
result = clf.predict(img)

# Dự đoán từ bytes (upload web)
with open("ảnh.jpg", "rb") as f:
    result = clf.predict_from_bytes(f.read())
```

---

## 5. Cấu Trúc Thư Mục

```
fruit-classification/
│
├── src/                          # 🔒 ML truyền thống (KHÔNG chỉnh sửa)
│   ├── preprocess.py             #    Pixel RGB 64×64 → 12,288 chiều
│   ├── train.py                  #    9 sklearn classifiers
│   ├── predict.py                #    FruitClassifier
│   └── ...
│
├── src_dl/                       # ✨ DL pipeline (nhánh mới)
│   ├── __init__.py               #    Package init
│   ├── backbone.py               #    CNNBackbone (MobileNetV3-Small)
│   ├── preprocess_dl.py          #    Trích xuất CNN embedding + cache
│   ├── train_dl.py               #    Huấn luyện sklearn trên embedding
│   └── predict_dl.py             #    FruitClassifierDL (inference)
│
├── data/
│   ├── raw/                      #    Ảnh gốc
│   └── processed/
│       ├── dataset/              #    Train/val/test (9 lớp)
│       │   ├── train/            #    140 ảnh/lớp
│       │   ├── val/              #    30 ảnh/lớp
│       │   └── test/             #    30 ảnh/lớp
│       ├── numeric/              #    Cache src/ (pixel vectors)
│       └── dl_features/          # ✨ Cache src_dl/ (CNN embeddings)
│           └── mobilenetv3_small/
│               ├── dl_dataset.npz   # Embeddings đã trích xuất
│               └── manifest.json    # Thông tin cache
│
├── models/                       # 🔒 Artifact của src/
├── models_dl/                    # ✨ Artifact của src_dl/
│   ├── classifier_dl.joblib      #    Pipeline sklearn tốt nhất
│   └── config.json               #    Cấu hình (feature_type, backbone, ...)
│
├── reports/                      # 🔒 Báo cáo của src/
├── reports_dl/                   # ✨ Báo cáo của src_dl/
│   └── summary_dl.txt            #    Kết quả 9 mô hình
│
├── docs/
│   └── cnn-feature-extraction.md #    Lý thuyết + đánh giá rủi ro CNN
│
├── requirements.txt              # 🔒 Phụ thuộc cơ bản
├── requirements_dl.txt           # ✨ Phụ thuộc bổ sung (torch, torchvision)
├── README.md                     # 🔒 Tài liệu chính (src/)
└── readme_dl.md                  # ✨ Tài liệu này (src_dl/)
```

> 🔒 = không thay đổi  |  ✨ = thêm mới trong nhánh DL

---

## 6. Tham Số & Tuỳ Chỉnh

### `preprocess_dl.py`

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--data-dir` | `data/processed/dataset` | Thư mục dataset |
| `--cache-dir` | `data/processed/dl_features/mobilenetv3_small` | Nơi lưu cache |
| `--backbone` | `mobilenetv3_small` | Backbone CNN |
| `--batch-size` | `32` | Số ảnh xử lý mỗi batch |
| `--seed` | `42` | Hạt giống ngẫu nhiên |
| `--force` | `False` | Trích xuất lại dù cache tồn tại |
| `--dataset-mode` | `dataset` | `dataset` hoặc `whole-scene` |

### `train_dl.py`

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--model-dir` | `models_dl` | Nơi lưu mô hình |
| `--report-dir` | `reports_dl` | Nơi lưu báo cáo |
| `--cache-dir` | `data/processed/dl_features/mobilenetv3_small` | Cache embedding |
| `--no-pca` | PCA bật | Tắt PCA |
| `--no-gridsearch` | GridSearch bật | Bỏ qua GridSearch |
| `--force-preprocess` | `False` | Trích xuất lại embedding |
| `--top-k` | `3` | Số mô hình tốt nhất lưu báo cáo |

### `predict_dl.py`

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `image` | — | Đường dẫn ảnh (bắt buộc) |
| `--model-dir` | `models_dl` | Thư mục mô hình |

---

## 7. Lý Thuyết: CNN Feature Extraction

### Transfer Learning

**Transfer Learning** là kỹ thuật tái sử dụng kiến thức đã học từ bài toán này (ImageNet) cho bài toán khác (phân loại trái cây).

Trong `src_dl/`, ta áp dụng chiến lược **Feature Extraction** (đóng băng backbone):

```
ImageNet (1.2M ảnh, 1000 lớp)
         ↓  Pretrain
MobileNetV3-Small weights → FROZEN (không cập nhật)
         ↓  Áp dụng
Ảnh trái cây → CNN embedding (576 chiều)
         ↓  Học mới
sklearn classifier (SVM, RF, LR, ...)
```

### Tại Sao MobileNetV3-Small?

| Backbone | Params | Embedding | Top-1 (ImageNet) | Phù hợp |
|---|---|---|---|---|
| ResNet-50 | 25.6M | 2048 | 76.1% | Quá nặng cho CPU |
| EfficientNet-B0 | 5.3M | 1280 | 77.1% | Nặng vừa |
| MobileNetV3-Small | **2.5M** | **576** | 67.7% | ✅ Nhẹ nhất, CPU-friendly |
| SqueezeNet | 1.2M | 512 | 58.1% | Quá yếu |

### Global Average Pooling (GAP)

Thay vì dùng toàn bộ feature map (H × W × C), ta lấy trung bình không gian:

```
Feature map: (1, 576, 7, 7)
     ↓ Adaptive Avg Pool
Embedding:   (1, 576, 1, 1)  →  flatten  →  (1, 576)
```

GAP giảm chiều dữ liệu mạnh mẽ trong khi vẫn giữ thông tin ngữ nghĩa tổng thể.

---

## 8. Đánh Giá Rủi Ro

| Rủi Ro | Mức Độ | Giải Pháp |
|---|---|---|
| **Domain shift (ImageNet → trái cây)** | Trung bình | GAP embedding vẫn capture màu sắc, kết cấu tốt; trái cây có màu đặc trưng rõ |
| **Phụ thuộc internet khi tải weights** | Thấp | Weights được cache sau lần đầu (~10 MB); kiểm tra `~/.cache/torch/` |
| **RAM/CPU không đủ** | Thấp | MobileNetV3-Small nhẹ nhất trong họ mobilenet; giảm `--batch-size` nếu thiếu RAM |
| **torch không tương thích Python 3.14** | Trung bình | Kiểm tra [PyTorch compatibility matrix](https://pytorch.org/get-started/locally/); thử Python 3.11/3.12 nếu gặp lỗi |
| **Overfitting (dữ liệu ít)** | Trung bình | Có augmentation × 2 bản train; PCA + regularization trong GridSearch |
| **Mất kiểm soát feature space** | Thấp-trung bình | Embedding được lưu dưới dạng `.npz` — có thể kiểm tra bằng t-SNE/UMAP |
| **Thời gian trích xuất lâu (CPU)** | Trung bình | Cache embedding một lần dùng mãi; chỉ cần preprocess lại khi dataset thay đổi |

---

## 9. Lưu Ý Quan Trọng

> **KHÔNG commit vào repository:**
> - Trọng số mô hình: `models_dl/*.joblib`
> - Cache embedding: `data/processed/dl_features/`
> - Báo cáo đánh giá: `reports_dl/`

Các thư mục này nên được thêm vào `.gitignore`:
```gitignore
models_dl/
reports_dl/
data/processed/dl_features/
```

> **KHÔNG tự cài torch:** Người dùng phải chủ động chạy:
> ```bash
> pip install -r requirements_dl.txt
> ```
> Nếu thiếu torch, `src_dl/` sẽ hiển thị thông báo lỗi rõ ràng bằng tiếng Việt.

> **Nhánh `src/` hoàn toàn độc lập:** `src_dl/` không import từ `src/predict.py`,
> `src/train.py`, `src/preprocess.py`. Hai nhánh có thể chạy song song.

---

*Tài liệu này được tạo tự động bởi AI agent. Phiên bản: 0.1.0*
