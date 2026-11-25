# Hướng dẫn huấn luyện mô hình (DL Pipeline)

Tài liệu chi tiết về quy trình huấn luyện mô hình Deep Learning phân loại trạng thái trái cây.

---

## 1. Cấu trúc bài toán và Dataset

- **Bài toán:** Phân loại đa lớp thành 9 lớp.
- **Định dạng dữ liệu:** Bộ classification theo thư mục tại `data/processed/dataset/{train,val,test}/`.
- **Taxonomy:** Cam/Chuối/Xoài × Chín/Chưa chín/Hỏng (nhãn: `cam-chin`, `chuoi-hong`, v.v.).
- **Split:** 1.260 ảnh train (140 ảnh/lớp), 270 validation, 270 test; tổng 1.800 ảnh (200/lớp), tỷ lệ 70% / 15% / 15%.

---

## 2. Tiền xử lý và trích xuất đặc trưng (`src.preprocess`)

Mỗi ảnh được resize về `224 × 224`, chuẩn hóa theo ImageNet (mean/std), đưa qua CNN backbone frozen để trích xuất vector embedding 576 chiều, lưu cache vào `data/processed/dl_features/`:

```powershell
python -m src.preprocess
```

Lệnh `src.train` tự động kiểm tra và chạy lại tiền xử lý nếu chưa có cache. Dùng `--force-preprocess` để buộc trích xuất lại.

---

## 3. Chạy huấn luyện

### Huấn luyện mô hình mặc định (MobileNetV3-Small, GPU/CPU tự động):

```powershell
python -m src.train --backbone mobilenetv3_small --epochs 15 --device auto
```

### So sánh cả 3 backbone (MobileNetV3-Small, MobileNetV2, ResNet-18):

```powershell
python -m src.train --backbone all --epochs 15 --device auto
```

### Tiếp tục huấn luyện từ checkpoint:

```powershell
python -m src.train --resume --epochs 10
```

### Các cờ tùy chọn:

- `--device {auto,cuda,cpu}`: Chỉ định phần cứng.
- `--lr 0.001`: Tốc độ học (Learning Rate).
- `--no-pca`: Tắt PCA, giữ toàn bộ chiều embedding.
- `--force-preprocess`: Trích xuất lại cache embedding.

---

## 4. Kết quả đầu ra

Sau khi huấn luyện, hệ thống tự động sinh:

- **`models/`**: `classifier.joblib` (pipeline scaler/PCA/MLP), `config.json` (metadata backbone, taxonomy, metrics).
- **`reports/`**:
  - `metrics.json`: Accuracy, Macro F1, Precision, Recall trên train/val/test.
  - `confusion_matrix.png`: Ma trận nhầm lẫn 9×9.
  - `roc_curve.png`: Đường ROC đa lớp kèm Macro AUC.
  - `loss_curve.png` / `training_history.png`: Diễn tiến loss và accuracy theo epoch.
  - `model_comparison.png`: So sánh 3 backbone.
  - `benchmark.csv`: FLOPs, số tham số, kích thước ONNX, độ trễ CPU/GPU.
