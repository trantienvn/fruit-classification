# Hướng dẫn huấn luyện mô hình (ML Pipeline)

Tài liệu chi tiết về quy trình chuyển đổi ảnh sang dữ liệu số và huấn luyện các mô hình Machine Learning truyền thống phân loại trạng thái trái cây.

---

## 1. Cấu trúc bài toán và Dataset

- **Bài toán:** Phân loại đa lớp thành 9 lớp.
- **Định dạng dữ liệu mặc định:** Bộ classification theo thư mục tại `data/processed/dataset/{train,val,test}/`.
- **Taxonomy mặc định:** Cam/Chuối/Xoài (`Orange`, `Banana`, `Mango`) × `Unripe`, `Ripe`, `Spoiled`.
- **Split hiện có:** 1.260 ảnh train, 271 validation, 270 test; giữ nguyên phân hoạch nguồn, tỷ lệ tổng thể xấp xỉ 70/15/15. (Pipeline cũng tự nhận diện `data/processed/whole-scene/` nếu truyền `--data-dir`).

---

## 2. Chuyển đổi ảnh sang dữ liệu dạng số (`src.preprocess`)

Thay vì trích xuất 90 đặc trưng thủ công, mỗi ảnh được chuyển sang hệ màu RGB, resize về `64 × 64` (`Image.LANCZOS`), chuẩn hóa giá trị pixel về `[0.0, 1.0]` kiểu `float32` và làm phẳng thành vector số `12.288` chiều (`64 × 64 × 3`).

Bước tiền xử lý loại ảnh trùng chính xác (ưu tiên giữ `test > val > train`) và lưu toàn bộ dữ liệu số vào `data/processed/numeric/numeric_dataset.npz` kèm `manifest.json`:

```powershell
python -m src.preprocess
```

Mỗi lần chạy `python -m src.train`, chương trình tự động kiểm tra hash dataset và nạp trực tiếp ma trận số từ `.npz` (chỉ chạy lại tiền xử lý nếu chưa có cache hoặc dataset ảnh thay đổi).

Toàn bộ `12.288` chiều pixel được chuẩn hóa bằng `StandardScaler` và giảm chiều bằng PCA (giữ 95% phương sai mặc định); scaler/PCA chỉ `fit` trên phần train của từng fold CV.

### Dùng dataset YOLO ảnh toàn cảnh (tùy chọn)

Nếu muốn chạy trên bộ YOLO đã xuất ảnh toàn cảnh (taxonomy `Apple/Banana/Mango`):

```powershell
python -m src.dataset --source-dir data/raw --output-dir data/processed/whole-scene
python -m src.preprocess --data-dir data/processed/whole-scene
python -m src.train --data-dir data/processed/whole-scene
```

Manifest ghi số ảnh giữ lại/loại theo split và class. Khi dùng `data/processed/dataset/`, model mặc định lưu vào `models/` và báo cáo vào `reports/`. Khi dùng `whole-scene`, model lưu vào `models-whole-scene/` và báo cáo vào `reports-whole-scene/`.

---

## 3. Các thuật toán so sánh

Pipeline so sánh 9 estimator: Dummy most-frequent, KNN, GaussianNB, Logistic
Regression, SVM RBF, SVM linear, Decision Tree, Random Forest và Gradient
Boosting. Các estimator hỗ trợ class weight dùng `balanced`.

---

## 4. Tối ưu siêu tham số (GridSearchCV)

Top-3 mô hình đạt Macro F1 cao nhất trên Validation được tinh chỉnh bằng
`GridSearchCV`, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
và `scoring='f1_macro'`. CV chỉ nhận train; scaler/PCA được fit riêng trong
từng fold.
- **KNN:** Tìm kiếm số láng giềng, weights và metric.
- **Logistic Regression:** Tìm kiếm $C$.
- **SVM RBF/linear:** Tìm kiếm $C$ và gamma với RBF.
- **Random Forest:** Tìm kiếm `n_estimators` $\in [100, 200, 300]$, `max_depth` $\in [\text{None}, 20, 30]$.
- **Gradient Boosting:** Tìm kiếm `learning_rate` $\in [0.05, 0.1]$, `max_depth` $\in [3, 5]$, `n_estimators` $\in [100, 200]$.

Mô hình có hiệu năng tốt nhất sau tinh chỉnh sẽ được chọn làm mô hình cuối cùng để đánh giá một lần duy nhất trên tập Test độc lập.

---

## 5. Hướng dẫn chạy huấn luyện

### Chạy huấn luyện mặc định (Cam/Chuối/Xoài, CPU)

```powershell
python -m src.preprocess
python -m src.train
```

### Chạy huấn luyện nhanh (bỏ qua GridSearchCV)

```powershell
python -m src.train --no-gridsearch
```

Pipeline so sánh 9 estimator scikit-learn trên CPU dựa trên dữ liệu pixel số
đã lưu trong `.npz`. Augmentation (xoay/lật/đổi sáng) chỉ áp dụng cho train
sau khi chọn model và được chuyển sang cùng dạng vector pixel chuẩn hóa `[0, 1]`;
validation, test và các fold CV không nhận ảnh tăng cường. Không có tùy chọn
CUDA/GPU hoặc backend PyTorch.

PCA 95% phương sai được bật mặc định. Để tắt và đối chiếu:

```powershell
python -m src.train --no-pca
```

## 6. Kết quả đầu ra

Sau khi chạy thành công, hệ thống tự động sinh:
- **Thư mục model:** Mặc định `models/` (hoặc `models-whole-scene/` với dataset whole-scene).
  - `classifier.joblib`: Pipeline tốt nhất (scaler/PCA/model).
  - `per-fruit/`: Pipeline riêng theo từng loại quả.
  - `scaler.joblib`: Bộ chuẩn hóa tương thích với artifact cũ.
  - `config.json`: Toàn bộ cấu hình metadata (`feature_type: "pixels_rgb_0_1"`, `crop_size: [64, 64]`, `n_features: 12288`), tên lớp, số mẫu và chỉ số metrics.
- **Thư mục báo cáo:** Mặc định `reports/` (hoặc `reports-whole-scene/`).
  - `confusion_matrix.png`: Ma trận nhầm lẫn trực quan trên tập test.
  - `roc_curves.png`: 9 đường cong ROC One-vs-Rest kèm giá trị AUC từng lớp.
  - `model_comparison.png` và `model_comparison.csv`: So sánh 9 estimator trên validation, gồm thời gian fit và inference/ảnh.
  - `per_fruit_comparison.csv`: So sánh model chung và model riêng theo quả.
  - `learning_curve.png`: Learning curve 5-fold chỉ trên train.
  - `learning_curve.csv`, `classification_metrics.csv`, `roc_auc.csv`, `summary.md`: số liệu theo điểm train, lớp, AUC và kết luận tự sinh từ lần chạy.
  - `misclassified/`: ảnh test dự đoán sai.
  - `feature_importance.png`: Top 20 thành phần/tọa độ quan trọng (nếu model hoặc PCA hỗ trợ).
  - `classification_report_<Model>.txt`: Bảng chi tiết Precision, Recall, F1 theo từng lớp (tên file theo model được chọn).

Web UI hiển thị các ảnh báo cáo trong mục **Thông tin mô hình → Biểu đồ đánh
giá**, gồm so sánh thuật toán, ma trận nhầm lẫn, ROC và feature importance khi
có. Ứng dụng mặc định dùng `models/` và `reports/`; `FRUIT_MODEL_DIR` và
`FRUIT_REPORTS_DIR` có thể chỉ định thư mục khác.
