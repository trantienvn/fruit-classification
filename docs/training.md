# Hướng dẫn huấn luyện mô hình (ML Pipeline)

Tài liệu chi tiết về quy trình nạp dữ liệu, trích xuất đặc trưng và huấn luyện các mô hình Machine Learning truyền thống phân loại độ chín của trái cây.

---

## 1. Cấu trúc bài toán và Dataset

- **Bài toán:** Phân loại đa lớp (Multi-class classification) thành **9 lớp kết hợp**:
  1. `Apple Unripe` (Táo chưa chín)
  2. `Apple Ripe` (Táo chín)
  3. `Apple Overripe` (Táo quá chín)
  4. `Banana Unripe` (Chuối chưa chín)
  5. `Banana Ripe` (Chuối chín)
  6. `Banana Overripe` (Chuối quá chín)
  7. `Mango Unripe` (Xoài chưa chín)
  8. `Mango Ripe` (Xoài chín)
  9. `Mango Overripe` (Xoài quá chín)
- **Định dạng dữ liệu:** YOLO Object Detection (tọa độ bounding box chuẩn hóa trong khoảng `[0, 1]`).
- **Quy mô dữ liệu thực tế:**
  - Tập Train: 1.559 ảnh (1.770 mẫu đối tượng sau khi crop)
  - Tập Validation: 445 ảnh (491 mẫu đối tượng)
  - Tập Test: 223 ảnh (246 mẫu đối tượng)
  - **Tổng cộng:** 2.227 ảnh gốc, 2.507 mẫu đối tượng trái cây.

---

## 2. Quy trình trích xuất 51 đặc trưng (Feature Extraction)

Mỗi đối tượng trái cây được cắt theo bounding box, chuyển sang RGB và co giãn về kích thước chuẩn `128 × 128` pixel. Vector đặc trưng gồm 51 chiều bao quát toàn diện các thuộc tính vật lý của trái cây:

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

### Chế độ ảnh toàn cảnh

Ngoài chế độ YOLO crop ở trên, có thể tạo bộ classification dùng nguyên ảnh
gốc. Pipeline giữ nguyên split train/val/test có sẵn; chỉ nhận ảnh mà tất cả
bounding box cùng mang một nhãn. Ảnh không có nhãn, nhãn sai định dạng hoặc
nhiều class bị loại vì không thể gán một nhãn classification duy nhất cho cảnh.
Ảnh được resize thành 128×128, cùng cách resize mà `FruitClassifier` dùng khi
inference.

```powershell
python -m src.dataset --source-dir data/raw --output-dir data/processed/whole-scene
python -m src.train --dataset-mode whole-scene --data-dir data/processed/whole-scene --no-gridsearch
```

Manifest ghi số ảnh giữ lại/loại theo split và class. Model được lưu mặc định
vào `models-whole-scene/`, báo cáo vào `reports-whole-scene/`; truyền
`--model-dir` nếu muốn chọn thư mục khác. Không dùng chung model directory giữa
hai chế độ và không resume TorchMLP từ một chế độ sang chế độ còn lại.

Model RandomForest train trên dataset hiện có đạt validation accuracy 94,81%
(Macro F1 0,9480), test accuracy 91,82% (Macro F1 0,9174) trên 220 ảnh test.
Đây là kết quả thực nghiệm của pipeline ảnh toàn cảnh; không so sánh trực tiếp
với metric YOLO-crop nếu tập ảnh/định nghĩa sample khác nhau.

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

### Chạy huấn luyện đầy đủ (Mặc định)

```powershell
python -m src.train --data-dir data/raw --model-dir models
```

### Chạy huấn luyện nhanh (Bỏ qua GridSearchCV)

```powershell
python -m src.train --data-dir data/raw --model-dir models --no-gridsearch
```

### Tùy chọn giảm chiều PCA

```powershell
python -m src.train --data-dir data/raw --model-dir models --pca
```

## 6. Huấn luyện tùy chọn trên GPU CUDA

Pipeline scikit-learn gồm 7 mô hình vẫn là mặc định và chạy trên CPU. Để chọn
GPU NVIDIA, dự án có backend PyTorch: nó huấn luyện một MLP trên cùng 51 đặc trưng
ảnh. PyTorch được khai báo trong `requirements.txt`; cài dependencies (hoặc cập nhật
môi trường hiện tại) bằng `uv pip install -r requirements.txt`. Wheel PyTorch đi
kèm CUDA runtime 12.6 để khớp driver 566.07 (CUDA 12.7) trên máy hiện tại; requirements
không cài driver NVIDIA. NVIDIA yêu cầu driver Windows 560.76 trở lên cho CUDA 12.6,
nên driver 566.07 đáp ứng điều kiện này. Wheel CUDA 12.6 có sẵn cho Windows/Python 3.14.
Nếu dùng máy khác, chọn wheel tương ứng trong [PyTorch Local](https://pytorch.org/get-started/locally/).

Kiểm tra PyTorch nhận CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA chưa sẵn sàng')"
```

Chạy huấn luyện vào thư mục riêng để giữ nguyên model scikit-learn trong `models/`:

```powershell
.\.venv\Scripts\python.exe -m src.train --device cuda --no-gridsearch --data-dir data/raw --model-dir models-gpu --epochs 100 --batch-size 64
```

`--device cuda` yêu cầu CUDA sẵn sàng; nếu không, chương trình báo lỗi thay vì
âm thầm chuyển sang CPU. MLP có early stopping dựa trên validation loss. Tùy chọn này huấn luyện riêng `TorchMLP`, không chạy 7
mô hình scikit-learn. Đọc ảnh, trích xuất 51 đặc trưng, chuẩn hóa và PCA (nếu bật)
vẫn chạy trên CPU; chỉ bước huấn luyện MLP chạy trên GPU. Tập dữ liệu hiện tại nhỏ,
nên GPU không nhất thiết nhanh hơn CPU.

Artifact trong `models-gpu/` có thể nạp bằng `FruitClassifier(model_dir="models-gpu")`.
Môi trường suy luận cần cài PyTorch; model dùng CUDA nếu có, nếu không sẽ suy luận
bằng CPU. Biểu đồ và classification report của lượt train CUDA được lưu riêng trong
`reports-gpu/`.

### Tiếp tục huấn luyện TorchMLP đã lưu

Trỏ `--model-dir` tới thư mục artifact PyTorch hiện có và thêm `--resume`. `--epochs`
là số epoch tối đa muốn chạy thêm; early stopping vẫn có thể dừng sớm. Chương trình
giữ trọng số tốt nhất đã lưu và dùng lại
scaler/PCA cũ để bảo đảm đầu vào tương thích. Chọn `--device cuda` để tiếp tục trên
GPU, hoặc `--device cpu` để chạy trên CPU:

```powershell
.\.venv\Scripts\python.exe -m src.train --resume --device cuda --no-gridsearch --data-dir data/raw --model-dir models-gpu --epochs 30 --batch-size 64
```

`--resume` chỉ hỗ trợ TorchMLP, không dùng cho model scikit-learn. Mỗi lượt tiếp tục
tạo optimizer Adam mới và chạy early stopping lại từ trọng số đã lưu. Kết quả ghi
đè artifact trong chính `--model-dir`; metadata lưu tổng epoch đã yêu cầu/đã train,
số epoch của lượt vừa chạy và trạng thái resume. Không cần truyền lại `--pca` nếu
artifact ban đầu đã dùng PCA.

---

## 7. Kết quả đầu ra

Sau khi chạy thành công, hệ thống tự động sinh:
- **Thư mục `models/`:**
  - `classifier.joblib`: Trọng số mô hình tốt nhất (SVM).
  - `scaler.joblib`: Bộ chuẩn hóa `StandardScaler` khớp với 51 features.
  - `config.json`: Toàn bộ cấu hình metadata, tên lớp, số mẫu và chỉ số metrics.
- **Thư mục `reports/`:**
  - `confusion_matrix.png`: Ma trận nhầm lẫn trực quan trên tập test.
  - `roc_curves.png`: 9 đường cong ROC One-vs-Rest kèm giá trị AUC từng lớp.
  - `model_comparison.png`: Biểu đồ cột so sánh Accuracy và Macro F1 của cả 7 mô hình.
  - `classification_report_SVM.txt`: Bảng chi tiết Precision, Recall, F1 theo từng lớp.
