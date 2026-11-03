# Hướng dẫn huấn luyện mô hình (ML Pipeline)

Tài liệu chi tiết về quy trình nạp dữ liệu, trích xuất đặc trưng và huấn luyện các mô hình Machine Learning truyền thống phân loại trạng thái trái cây.

---

## 1. Cấu trúc bài toán và Dataset

- **Bài toán:** Phân loại đa lớp thành 9 lớp; nhãn tùy theo dataset mode.
- **Định dạng dữ liệu mặc định:** Ảnh nguyên cảnh tại `data/processed/whole-scene/{train,val,test}/`; giữ nguyên ảnh, không crop bounding box.
- **Taxonomy mặc định:** Apple/Banana/Mango × Unripe/Ripe/Overripe từ dataset YOLO cũ trong `data/raw/`.
- **Dataset classification khác:** Có thể chọn rõ mode `folder` cho 1.260 ảnh train, 270 validation và 270 test (Cam/Chuối/Xoài × Unripe/Ripe/Spoiled) tại `data/processed/dataset/`; mỗi lớp có 140/30/30 ảnh tương ứng.

---

## 2. Quy trình trích xuất 51 đặc trưng (Feature Extraction)

Mỗi ảnh được chuyển sang RGB và co giãn về kích thước chuẩn `128 × 128` pixel. Vector đặc trưng gồm 51 chiều bao quát các thuộc tính thị giác của trái cây:

1. **Đặc trưng màu sắc (Color Features - 33 chiều):**
   - *RGB Color Histogram:* 8 bins trên mỗi kênh R, G, B $\rightarrow$ 24 chiều.
   - *RGB Moments:* Trung bình và độ lệch chuẩn của các kênh R, G, B $\rightarrow$ 6 chiều.
   - *HSV Statistics:* Trung bình Hue (H), trung bình Saturation (S) và độ lệch chuẩn Saturation (S) $\rightarrow$ 3 chiều.
2. **Đặc trưng kết cấu bề mặt (Texture Features - 14 chiều):**
   - *Local Binary Pattern (LBP):* Bán kính $R=1$, số điểm lân cận $P=8$, phương pháp uniform histogram $\rightarrow$ 10 chiều.
   - *Gray-Level Co-occurrence Matrix (GLCM):* Tính độ tương phản (contrast), độ khác biệt (dissimilarity), độ đồng nhất (homogeneity) và năng lượng (energy) ở khoảng cách $d=1$, góc $0^\circ$ $\rightarrow$ 4 chiều.
3. **Đặc trưng hình dạng (Shape Features - 4 chiều):**
   - Phân đoạn nhị phân bằng Otsu thresholding trên ảnh mức xám.
   - Trích xuất: Tỷ lệ diện tích contour trên toàn khung hình (Area ratio), tỷ lệ rộng/cao của bounding box (Aspect ratio), độ đặc trong bounding box (Extent) và đường kính tương đương đã chuẩn hóa (Equivalent diameter) $\rightarrow$ 4 chiều.

Toàn bộ 51 đặc trưng được chuẩn hóa bằng `StandardScaler` (chỉ fit trên tập train, sau đó transform cho val và test).

### Dataset mặc định: ảnh toàn cảnh từ YOLO

Dataset YOLO trong `data/raw/` được chuyển thành bộ classification dùng nguyên
ảnh gốc. Pipeline giữ nguyên split train/val/test có sẵn; chỉ nhận ảnh mà tất cả
bounding box cùng mang một nhãn. Ảnh không có nhãn, nhãn sai định dạng hoặc
nhiều class bị loại vì không thể gán một nhãn classification duy nhất cho cảnh.
Ảnh được resize thành 128×128, cùng cách resize mà `FruitClassifier` dùng khi
inference. Đây là chế độ mặc định của lệnh train.

```powershell
python -m src.dataset --source-dir data/raw --output-dir data/processed/whole-scene
python -m src.train
```

Manifest ghi số ảnh giữ lại/loại theo split và class. Model được lưu mặc định
vào `models-whole-scene/`, báo cáo vào `reports-whole-scene/`; truyền
`--model-dir` nếu muốn chọn thư mục khác. Không dùng chung model directory giữa
hai chế độ.

Artifact của chế độ này vẫn dùng taxonomy Apple/Banana/Mango cũ và được lưu
riêng trong `models-whole-scene/`. Muốn dùng taxonomy Cam/Chuối/Xoài, chọn rõ
`--dataset-mode folder --data-dir data/processed/dataset --model-dir models`.

---

## 3. Các thuật toán so sánh

Pipeline khảo sát 7 thuật toán học máy truyền thống:
1. **K-Nearest Neighbors (KNN):** $k=5$, Euclidean distance.
2. **Naive Bayes (GaussianNB):** Phân phối chuẩn giả định độc lập.
3. **Logistic Regression (LR):** Đa thức (multinomial), tối ưu L-BFGS, $C=1.0$.
4. **Support Vector Machine (SVM):** Kernel RBF, $C=10 \rightarrow 50$, `probability=True`.
5. **Decision Tree (DT):** CART algorithm, `max_depth=15`.
6. **Random Forest (RF):** 200 cây quyết định, bagging kết hợp ngẫu nhiên đặc trưng.
7. **Gradient Boosting (GBM):** Boosting tuần tự tối ưu hàm mất mát log-loss.

---

## 4. Tối ưu siêu tham số (GridSearchCV)

Top-3 mô hình đạt Macro F1 cao nhất trên tập Validation được đưa vào tinh chỉnh bằng `GridSearchCV` với phân hoạch cố định `PredefinedSplit(train vs val)`:
- **SVM:** Tìm kiếm $C \in [1, 10, 50]$, $\gamma \in [\text{'scale'}, \text{'auto'}]$.
- **Random Forest:** Tìm kiếm `n_estimators` $\in [100, 200, 300]$, `max_depth` $\in [\text{None}, 20, 30]$.
- **Gradient Boosting:** Tìm kiếm `learning_rate` $\in [0.05, 0.1]$, `max_depth` $\in [3, 5]$, `n_estimators` $\in [100, 200]$.

Mô hình có hiệu năng tốt nhất sau tinh chỉnh sẽ được chọn làm mô hình cuối cùng để đánh giá một lần duy nhất trên tập Test độc lập.

---

## 5. Hướng dẫn chạy huấn luyện

### Chạy huấn luyện mặc định (ảnh nguyên cảnh, CPU)

```powershell
python -m src.train
```

### Chạy huấn luyện nhanh (bỏ qua GridSearchCV)

```powershell
python -m src.train --no-gridsearch
```

Pipeline luôn huấn luyện 7 mô hình scikit-learn trên CPU. Không có tùy chọn
CUDA/GPU hoặc backend PyTorch trong lệnh train. Có thể chọn dataset classification
Cam/Chuối/Xoài bằng cách chỉ rõ mode và đường dẫn:

```powershell
python -m src.train --dataset-mode folder --data-dir data/processed/dataset --model-dir models
```

Để bật PCA giảm chiều cho dataset mặc định:

```powershell
python -m src.train --pca
```

## 6. Kết quả đầu ra

Sau khi chạy thành công, hệ thống tự động sinh:
- **Thư mục model:** Mặc định `models-whole-scene/`; mode `folder` mặc định dùng `models/`.
  - `classifier.joblib`: Trọng số mô hình tốt nhất (SVM).
  - `scaler.joblib`: Bộ chuẩn hóa `StandardScaler` khớp với 51 features.
  - `config.json`: Toàn bộ cấu hình metadata, tên lớp, số mẫu và chỉ số metrics.
- **Thư mục báo cáo:** Mặc định `reports-whole-scene/`; mode `folder` dùng `reports/`.
  - `confusion_matrix.png`: Ma trận nhầm lẫn trực quan trên tập test.
  - `roc_curves.png`: 9 đường cong ROC One-vs-Rest kèm giá trị AUC từng lớp.
  - `model_comparison.png`: Biểu đồ cột so sánh Accuracy và Macro F1 của cả 7 mô hình.
  - `feature_importance.png`: Top 20 đặc trưng quan trọng. Dùng importance/|coef| tích hợp nếu model hỗ trợ; nếu không, dùng permutation importance trên validation (mức giảm Macro F1 sau khi xáo trộn từng đặc trưng).
  - `classification_report_SVM.txt`: Bảng chi tiết Precision, Recall, F1 theo từng lớp.

Web UI hiển thị các ảnh báo cáo trong mục **Thông tin mô hình → Biểu đồ đánh
giá**, gồm so sánh thuật toán, ma trận nhầm lẫn, ROC và feature importance khi
có. Ứng dụng chọn thư mục báo cáo theo `dataset_mode` trong `config.json`;
`FRUIT_REPORTS_DIR` có thể dùng để chỉ định thư mục khác.
