# Hướng dẫn huấn luyện mô hình (ML Pipeline)

Tài liệu chi tiết về quy trình nạp dữ liệu, trích xuất đặc trưng và huấn luyện các mô hình Machine Learning truyền thống phân loại trạng thái trái cây.

---

## 1. Cấu trúc bài toán và Dataset

- **Bài toán:** Phân loại đa lớp thành 9 lớp.
- **Định dạng dữ liệu:** Ảnh nguyên cảnh theo split tại `data/processed/whole-scene/{train,val,test}/`.
- **Taxonomy:** Apple/Banana/Mango × Unripe/Ripe/Overripe.
- **Split hiện có:** 1.260 ảnh train, 271 validation, 270 test; giữ nguyên phân hoạch nguồn, tỷ lệ tổng thể xấp xỉ 70/15/15.

---

## 2. Quy trình trích xuất 90 đặc trưng (Feature Extraction)

Mỗi ảnh được chuyển RGB, resize về `128 × 128`, median blur và phân đoạn bằng Otsu trên S/xám, morphology; adaptive threshold là fallback. Đặc trưng màu và kết cấu chỉ tính trong mask vùng quả:

1. **Màu sắc (70 chiều):** RGB và HSV histogram, mỗi kênh có 8 bins; mean/std của RGB, HSV và Lab; tỷ lệ xanh, vàng, cam và nâu/đen.
2. **Kết cấu (14 chiều):**
   - *Local Binary Pattern (LBP):* Bán kính $R=1$, số điểm lân cận $P=8$, phương pháp uniform histogram $\rightarrow$ 10 chiều.
   - *GLCM:* contrast, homogeneity, energy, correlation ở khoảng cách $d=1$, góc $0^\circ$ $\rightarrow$ 4 chiều.
3. **Hình dạng (6 chiều):** Tỷ lệ diện tích, chu vi chuẩn hóa, tỷ lệ rộng/cao, aspect ratio và circularity từ contour mask.

Toàn bộ 90 đặc trưng được chuẩn hóa trong scikit-learn Pipeline; scaler/PCA chỉ fit trên phần train của từng fold CV.

### Chuẩn bị dữ liệu nguyên cảnh

Dataset YOLO trong `data/raw/` được chuyển thành bộ classification dùng nguyên
ảnh gốc. Pipeline giữ nguyên split train/val/test có sẵn; chỉ nhận ảnh mà tất cả
bounding box cùng mang một nhãn. Ảnh không có nhãn, nhãn sai định dạng hoặc
nhiều class bị loại vì không thể gán một nhãn classification duy nhất cho cảnh.
Ảnh được resize thành 128×128, cùng cách resize mà `FruitClassifier` dùng khi
inference. Đây là dữ liệu duy nhất được pipeline train hỗ trợ.

```powershell
python -m src.dataset --source-dir data/raw --output-dir data/processed/whole-scene
python -m src.train
```

Manifest ghi số ảnh giữ lại/loại theo split và class. Model được lưu mặc định
vào `models-whole-scene/`, báo cáo vào `reports-whole-scene/`; truyền
`--model-dir` nếu muốn chọn thư mục khác. Không dùng chung model directory giữa
hai chế độ.

Artifact dùng taxonomy Apple/Banana/Mango và được lưu trong
`models-whole-scene/`.

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

### Chạy huấn luyện mặc định (Apple/Banana/Mango, CPU)

```powershell
python -m src.train
```

### Chạy huấn luyện nhanh (bỏ qua GridSearchCV)

```powershell
python -m src.train --no-gridsearch
```

Pipeline so sánh 9 estimator scikit-learn trên CPU. Augmentation (xoay/lật/đổi
sáng) chỉ áp dụng cho train sau khi chọn model; validation, test và các fold CV
không nhận ảnh tăng cường. Không có tùy chọn CUDA/GPU hoặc backend PyTorch.

```powershell
python -m src.dataset --source-dir data/raw --output-dir data/processed/whole-scene
python -m src.train
```

PCA 95% phương sai được bật mặc định. Để tắt và đối chiếu:

```powershell
python -m src.train --no-pca
```

## 6. Kết quả đầu ra

Sau khi chạy thành công, hệ thống tự động sinh:
- **Thư mục model:** Mặc định `models-whole-scene/`.
  - `classifier.joblib`: Pipeline tốt nhất (scaler/PCA/model).
  - `per-fruit/`: Pipeline riêng theo từng loại quả.
  - `scaler.joblib`: Bộ chuẩn hóa tương thích với artifact cũ.
  - `config.json`: Toàn bộ cấu hình metadata, tên lớp, số mẫu và chỉ số metrics.
- **Thư mục báo cáo:** Mặc định `reports-whole-scene/`.
  - `confusion_matrix.png`: Ma trận nhầm lẫn trực quan trên tập test.
  - `roc_curves.png`: 9 đường cong ROC One-vs-Rest kèm giá trị AUC từng lớp.
  - `model_comparison.png` và `model_comparison.csv`: So sánh 9 estimator trên validation, gồm thời gian fit và inference/ảnh.
  - `ablation_comparison.csv` và `ablation_comparison.png`: Màu; màu+kết cấu; màu+kết cấu+hình dạng.
  - `per_fruit_comparison.csv`: So sánh model chung và model riêng theo quả.
  - `learning_curve.png`: Learning curve 5-fold chỉ trên train.
  - `learning_curve.csv`, `classification_metrics.csv`, `roc_auc.csv`, `summary.md`: số liệu theo điểm train, lớp, AUC và kết luận tự sinh từ lần chạy.
  - `mask_samples/` và `misclassified/`: mask train mẫu và ảnh test dự đoán sai.
  - `feature_importance.png`: Top 20 đặc trưng quan trọng. Dùng importance/|coef| tích hợp nếu model hỗ trợ; nếu không, dùng permutation importance trên validation (mức giảm Macro F1 sau khi xáo trộn từng đặc trưng).
  - `classification_report_RandomForest.txt`: Bảng chi tiết Precision, Recall, F1 theo từng lớp (tên file theo model được chọn).

Web UI hiển thị các ảnh báo cáo trong mục **Thông tin mô hình → Biểu đồ đánh
giá**, gồm so sánh thuật toán, ma trận nhầm lẫn, ROC và feature importance khi
có. Ứng dụng dùng `reports-whole-scene/`; `FRUIT_REPORTS_DIR` có thể chỉ định
thư mục báo cáo khác.
