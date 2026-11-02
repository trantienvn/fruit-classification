# Báo cáo kết quả thực nghiệm mô hình

## Kết quả hiện hành — dataset Cam/Chuối/Xoài

Dataset classification tại `data/processed/dataset/` có 1.270 ảnh train,
361 validation và 184 test. GridSearch chọn SVM với validation accuracy
94,46%, Macro F1 0,9448. Trên test độc lập, SVM đạt accuracy 91,30%,
Macro F1 0,9115 và Macro AUC 0,9940.

| Lớp | Precision | Recall | F1 | Số mẫu |
|---|---:|---:|---:|---:|
| Orange Unripe | 0,91 | 1,00 | 0,95 | 21 |
| Orange Ripe | 0,95 | 0,95 | 0,95 | 20 |
| Orange Spoiled | 0,86 | 0,95 | 0,90 | 20 |
| Banana Unripe | 0,86 | 0,95 | 0,90 | 20 |
| Banana Ripe | 1,00 | 0,82 | 0,90 | 22 |
| Banana Spoiled | 1,00 | 1,00 | 1,00 | 20 |
| Mango Unripe | 0,90 | 0,90 | 0,90 | 20 |
| Mango Ripe | 0,91 | 1,00 | 0,95 | 20 |
| Mango Spoiled | 0,82 | 0,67 | 0,74 | 21 |

Báo cáo chi tiết và biểu đồ mới nằm trong `reports/`; cấu hình model ghi tại
`models/config.json` (artifact model bị Git bỏ qua).

## Kết quả lịch sử — dataset YOLO Apple/Banana/Mango

Các mục bên dưới ghi lại thí nghiệm cũ trên bộ **Fruit Ripeness** (Huzaifa /
Roboflow Universe), không phải kết quả của model hiện hành.

---

## 1. Dữ liệu thực nghiệm

- **Tổng số mẫu đối tượng gán nhãn:** 2.507 mẫu (cắt từ 2.227 ảnh theo bounding box YOLO).
- **Phân hoạch:**
  - Tập Huấn luyện (Train): 1.770 mẫu (70.6%)
  - Tập Kiểm định (Validation): 491 mẫu (19.6%)
  - Tập Kiểm thử (Test): 246 mẫu (9.8%)
- **Số lượng đặc trưng trích xuất:** 51 đặc trưng (33 Màu sắc RGB/HSV + 14 Kết cấu LBP/GLCM + 4 Hình dạng Otsu).

---

## 2. So sánh hiệu năng 7 mô hình Machine Learning (Tập Validation)

| Thuật toán | Accuracy | Macro F1 | Thời gian Train (s) | Thời gian Inference (ms) |
|---|---:|---:|---:|---:|
| **Support Vector Machine (SVM)** | **0.9491** | **0.9494** | 0.3 | 0.08 |
| **Random Forest (RF)** | 0.9430 | 0.9441 | 0.3 | 0.05 |
| **Gradient Boosting (GBM)** | 0.9287 | 0.9316 | 12.8 | 0.02 |
| **K-Nearest Neighbors (KNN)** | 0.8961 | 0.8976 | 0.0 | 0.04 |
| **Logistic Regression (LR)** | 0.8473 | 0.8455 | 0.1 | 0.01 |
| **Decision Tree (DT)** | 0.8330 | 0.8365 | 0.0 | 0.01 |
| **Naive Bayes (GaussianNB)** | 0.6782 | 0.6714 | 0.0 | 0.01 |

*Nhận xét:*
- Nhóm mô hình dựa trên đường bao phân cách phi tuyến (SVM RBF) và tập hợp cây quyết định (Random Forest, Gradient Boosting) đạt hiệu năng vượt trội ($F1 > 93\%$).
- SVM kernel RBF cho kết quả ban đầu tốt nhất trên tập Validation ($F1 = 94.94\%$).
- Naive Bayes đạt hiệu năng thấp nhất do giả định tính độc lập có điều kiện giữa các đặc trưng màu sắc và kết cấu không thỏa mãn hoàn toàn trong thực tế.

---

## 3. Kết quả tối ưu siêu tham số (GridSearchCV trên Top-3 Mô hình)

Áp dụng `GridSearchCV` trên 3 mô hình dẫn đầu với phân hoạch cố định:

| Mô hình | Siêu tham số tối ưu | Val F1 ban đầu | Val F1 sau tinh chỉnh | Mức cải thiện |
|---|---|---:|---:|---:|
| **SVM** | `C=50`, `gamma='scale'`, `kernel='rbf'` | 0.9494 | **0.9546** | **+0.52%** |
| **Random Forest** | `n_estimators=200`, `max_depth=20` | 0.9441 | 0.9459 | +0.18% |
| **Gradient Boosting**| `learning_rate=0.1`, `max_depth=5`, `n_estimators=200` | 0.9316 | 0.9443 | +1.27% |

$\rightarrow$ **Mô hình được lựa chọn:** **Support Vector Machine (SVM Tuned)** với $C=50$, đạt $F1 = 95.46\%$ trên tập Validation.

---

## 4. Kết quả đánh giá cuối cùng trên tập Test độc lập

Mô hình SVM sau khi tối ưu được đánh giá trên 246 mẫu của tập Test độc lập (hoàn toàn không tham gia vào quá trình huấn luyện hay tinh chỉnh tham số).

### Các chỉ số tổng quát:
- **Độ chính xác tổng thể (Accuracy):** **93.50%**
- **Macro F1-Score:** **0.9371**
- **Weighted F1-Score:** **0.9372**
- **Độ đo ROC-AUC trung bình đa lớp (Macro AUC):** **0.9985**

### Báo cáo chi tiết theo từng lớp (Classification Report):

| Lớp đối tượng | Precision | Recall | F1-Score | Số mẫu (Support) |
|---|---:|---:|---:|---:|
| **Apple Unripe** | 0.90 | 0.96 | 0.93 | 28 |
| **Apple Ripe** | 1.00 | 0.93 | 0.97 | 30 |
| **Apple Overripe** | 1.00 | 0.88 | 0.94 | 25 |
| **Banana Unripe** | 0.96 | 0.93 | 0.95 | 28 |
| **Banana Ripe** | 0.81 | 0.97 | 0.88 | 31 |
| **Banana Overripe** | 0.92 | 0.92 | 0.92 | 24 |
| **Mango Unripe** | 0.96 | 0.96 | 0.96 | 25 |
| **Mango Ripe** | 0.93 | 0.87 | 0.90 | 30 |
| **Mango Overripe** | 1.00 | 1.00 | 1.00 | 25 |
| **Trung bình Macro** | **0.94** | **0.94** | **0.94** | **246** |
| **Trung bình có trọng số** | **0.94** | **0.93** | **0.94** | **246** |

---

## 5. Phân tích kết quả thực nghiệm

1. **Khả năng phân biệt loại trái cây:**
   - Mô hình phân biệt gần như tuyệt đối giữa 3 loại trái cây (Táo, Chuối, Xoài) nhờ sự kết hợp giữa histogram màu sắc và hình dạng tỷ lệ khung hình.
   - Lớp `Mango Overripe` và `Apple Ripe` đạt độ chính xác Precision = 100%.

2. **Khả năng phân biệt độ chín:**
   - Sự dịch chuyển màu sắc từ xanh lục (chưa chín) sang vàng/đỏ (chín) và xuất hiện đốm nâu/vùng tối (quá chín) được mô tả rất tốt thông qua kết hợp Histogram màu, giá trị trung bình HSV và đặc trưng kết cấu LBP/GLCM.
   - Một số ít trường hợp nhầm lẫn xuất hiện giữa `Banana Ripe` và `Banana Overripe` hoặc `Banana Unripe` do chuối trong giai đoạn đầu chín có thể có cả vùng xanh và vàng đồng thời trên vỏ.

3. **Tốc độ xử lý:**
   - Trích xuất đặc trưng: ~5–10 ms mỗi ảnh.
   - Suy luận SVM: < 0.1 ms mỗi mẫu.
   - Tổng thời gian phản hồi toàn trình (từ đọc ảnh đến trả về nhãn và xác suất): ~10–15 ms, hoàn toàn đáp ứng yêu cầu xử lý thời gian thực.
