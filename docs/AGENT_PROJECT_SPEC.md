# PROJECT SPECIFICATION — PHÂN LOẠI CHẤT LƯỢNG VÀ ĐỘ CHÍN CỦA TRÁI CÂY

> Tài liệu hướng dẫn coding agent triển khai dự án dựa trên báo cáo Word của **Nhóm 03 – Học phần Học máy, năm 2026**.  
> Giáo viên hướng dẫn: **TS. Ngô Hữu Huy**.  
> Thành viên: **Trần Văn Tiến, Đặng Xuân Thảo, Lý Văn Quyến**.

---

## 1. Vai trò và nguyên tắc làm việc của coding agent

Bạn là coding agent chịu trách nhiệm khảo sát mã nguồn hiện có, triển khai, kiểm thử và hoàn thiện chương trình phục vụ đề tài:

**Phân loại chất lượng và độ chín của trái cây.**

### 1.1. Nguyên tắc bắt buộc

1. **Bắt đầu bằng việc đọc repository hiện tại**, đặc biệt là thư mục `docs/`, README, cấu hình, mã nguồn, dữ liệu và các kết quả huấn luyện đã có.
2. Không tự ý tạo một dự án mới hoặc thay đổi công nghệ đang sử dụng nếu repository đã có cấu trúc và lựa chọn công nghệ.
3. Không xóa, ghi đè hoặc di chuyển dữ liệu, model, notebook, báo cáo hay mã nguồn hiện có nếu chưa xác định rõ tác động.
4. Không tự tạo dataset, nhãn, kết quả accuracy, confusion matrix, ROC/AUC hoặc số liệu thực nghiệm giả.
5. Không khẳng định mô hình đã hỗ trợ một loại trái cây hay một nhãn nếu dataset và mã nguồn không chứng minh điều đó.
6. Phân biệt rõ:
   - Nội dung đã được xác nhận trong báo cáo.
   - Nội dung đã có trong repository.
   - Nội dung agent đề xuất nhưng chưa được người dùng xác nhận.
7. Khi thiếu thông tin quan trọng, trước tiên kiểm tra repository và tài liệu liên quan. Chỉ hỏi người dùng khi không thể xác minh bằng các nguồn hiện có.
8. Ưu tiên thay đổi nhỏ, có thể kiểm thử và tương thích với cấu trúc hiện tại.
9. Sau mỗi nhóm thay đổi, chạy kiểm thử phù hợp và báo cáo chính xác kết quả.
10. Ghi lại các quyết định kỹ thuật, cách cài đặt và cách chạy chương trình trong tài liệu `docs/`.

### 1.2. Quy trình làm việc

Thực hiện theo thứ tự:

1. Khảo sát cấu trúc thư mục và trạng thái Git.
2. Đọc tài liệu trong `docs/` và README.
3. Xác định ngôn ngữ, framework, thư viện, entry point và cách chạy hiện tại.
4. Kiểm tra dataset, nhãn, cách tổ chức dữ liệu và các model có sẵn.
5. Đối chiếu hiện trạng với tài liệu đặc tả này.
6. Lập kế hoạch triển khai theo từng phần nhỏ.
7. Thực hiện thay đổi.
8. Chạy lint/test/kiểm thử thực tế phù hợp với dự án.
9. Cập nhật tài liệu và tổng kết những phần đã hoàn thành, chưa hoàn thành, còn giả định.

---

## 2. Tổng quan đề tài

### 2.1. Mục tiêu

Xây dựng chương trình sử dụng các phương pháp học máy để phân loại:

- **Độ chín của trái cây**.
- **Chất lượng của trái cây**.

Đầu vào dự kiến là ảnh trái cây. Hệ thống cần xử lý ảnh, trích xuất đặc trưng, đưa đặc trưng vào mô hình học máy và trả về kết quả phân loại.

### 2.2. Phạm vi phương pháp trong báo cáo

Báo cáo định hướng phương pháp học máy truyền thống kết hợp xử lý ảnh và trích xuất đặc trưng thủ công:

- Tiền xử lý ảnh.
- Chuyển đổi không gian màu.
- Tách nền/phân đoạn vùng trái cây.
- Trích xuất đặc trưng màu sắc, kết cấu và hình dạng.
- Chuẩn hóa đặc trưng và xem xét giảm chiều bằng PCA.
- Huấn luyện, tinh chỉnh và so sánh nhiều mô hình học máy.
- Đánh giá bằng các chỉ số phân loại.
- Đóng gói pipeline và xây dựng giao diện nhập ảnh, hiển thị kết quả.

**Không tự ý thay thế toàn bộ hướng tiếp cận này bằng CNN, YOLO, Vision Transformer hoặc một mô hình deep learning khác.** Báo cáo có nhắc CNN/Transformer trong phần lý thuyết tổng quan, nhưng phần phương pháp đề xuất tập trung vào đặc trưng ảnh và các thuật toán học máy như KNN, Logistic Regression, SVM, Random Forest, Gradient Boosting và Naive Bayes. Nếu muốn đổi hướng, cần có yêu cầu hoặc xác nhận của người dùng.

### 2.3. Đầu vào và đầu ra dự kiến

| Thành phần | Nội dung |
|---|---|
| Đầu vào | Ảnh trái cây do người dùng cung cấp |
| Xử lý | Tiền xử lý ảnh, phân đoạn, trích xuất đặc trưng, chuẩn hóa và dự đoán |
| Đầu ra | Kết quả phân loại độ chín và chất lượng theo những nhãn thực sự có trong dataset/model |
| Giao diện | Cho phép nhập/chọn ảnh, hiển thị ảnh đầu vào và kết quả phân loại |

Báo cáo chưa quy định định dạng ảnh, kích thước ảnh, API, giao thức trao đổi dữ liệu hoặc framework giao diện. Agent phải kiểm tra dự án hiện tại trước khi quyết định.

---

## 3. Định nghĩa bài toán và nhãn

### 3.1. Phân loại độ chín

Báo cáo nêu ba mức độ chín minh họa:

| Nhãn | Tên tiếng Anh | Diễn giải |
|---|---|---|
| Chưa chín | `Unripe` | Trái cây chưa đạt độ chín ăn được |
| Chín | `Ripe` | Trái cây đạt trạng thái chín |
| Quá chín | `Overripe` | Trái cây đã chín quá mức |

Báo cáo có nhắc ví dụ chuối, táo và xoài. Đây là ví dụ trong phần lý thuyết, **không mặc định rằng dataset thực tế chỉ có hoặc chắc chắn có ba loại này**.

### 3.2. Phân loại chất lượng

Báo cáo trình bày tiêu chí chất lượng và phân hạng thương mại:

- **Hạng Đặc biệt (Extra Class):** chất lượng hoàn hảo, đồng đều, không có khuyết tật ngoại quan.
- **Hạng I (Class 1):** chất lượng tốt, độ đồng đều cao; cho phép lỗi nhẹ trên vỏ nếu không ảnh hưởng thẩm mỹ chung.
- **Hạng II (Class 2):** có khuyết tật rõ hơn về hình dáng hoặc màu sắc, nhưng phần thịt quả vẫn ngon và an toàn.
- **Hạng chế biến:** không đạt chuẩn ăn tươi, có thể dùng để chế biến.

Đây là các tiêu chí được trình bày trong báo cáo. Chưa có bằng chứng cho thấy dataset đã được gán nhãn theo đúng bốn hạng này. Agent phải kiểm tra nhãn hiện có và không tự ánh xạ nhãn nếu chưa có căn cứ.

### 3.3. Hai thuộc tính cần phân loại

Báo cáo đặt mục tiêu xử lý đồng thời độ chín và chất lượng, nhưng chưa chốt cách triển khai kỹ thuật. Cần khảo sát dữ liệu để xác định một trong các phương án:

- Hai mô hình độc lập: một mô hình dự đoán độ chín, một mô hình dự đoán chất lượng.
- Một mô hình đa đầu ra/multi-output.
- Một mô hình phân loại nhãn kết hợp nếu dataset thực sự có nhãn kết hợp.

Không tự chọn phương án trước khi kiểm tra cấu trúc nhãn và số lượng mẫu. Ghi rõ ưu/nhược điểm của phương án được chọn.

---

## 4. Cơ sở xử lý ảnh

### 4.1. Biểu diễn và không gian màu

Báo cáo đề cập:

- **RGB:** ba kênh Red, Green, Blue; phù hợp cho thiết bị hiển thị, nhưng thông tin màu và độ sáng bị trộn lẫn.
- **HSV:** gồm Hue, Saturation, Value; thuận tiện cho xử lý màu vì tách sắc độ/độ bão hòa khỏi độ sáng.
- **Lab:** gồm Lightness và hai trục màu `a`, `b`; hữu ích cho so sánh màu và hiệu chỉnh màu.

Agent cần giữ nhất quán thứ tự kênh, kiểu dữ liệu và miền giá trị khi chuyển đổi/trích xuất đặc trưng.

### 4.2. Lọc nhiễu

Các phương pháp được báo cáo đề cập:

- **Mean filter:** thay pixel bằng trung bình các pixel lân cận; giảm nhiễu nhưng có thể làm mờ biên.
- **Median filter:** chọn trung vị trong vùng lân cận; phù hợp để loại nhiễu muối tiêu và giữ biên tốt hơn.
- **Gaussian filter:** làm mịn dựa trên phân phối Gauss.

Không áp dụng mọi bộ lọc một cách máy móc. Chọn phương pháp dựa trên đặc điểm dữ liệu và đánh giá ảnh trước/sau xử lý.

### 4.3. Cân bằng histogram

Mục đích: cải thiện độ tương phản, làm rõ chi tiết ở vùng tối/sáng. Báo cáo mô tả các bước tính histogram, CDF và ánh xạ mức xám.

Chỉ áp dụng khi cần thiết; tránh làm biến đổi màu sắc quan trọng đối với việc nhận biết độ chín.

### 4.4. Phân đoạn và tách nền

Báo cáo đề cập ngưỡng Otsu:

1. Chuyển ảnh màu sang ảnh xám.
2. Tính histogram mức xám.
3. Tìm ngưỡng tự động bằng Otsu.
4. Tạo ảnh nhị phân để tách tiền cảnh và hậu cảnh.

Ưu điểm: nhanh, tự động, phù hợp khi histogram có hai đỉnh rõ. Hạn chế: dễ lỗi khi ánh sáng không đều hoặc histogram không tách biệt rõ. Có thể xem xét Adaptive Thresholding trong trường hợp phù hợp.

Agent cần kiểm tra kết quả mask trên dữ liệu thực tế; không mặc định Otsu luôn tách được trái cây khỏi nền.

---

## 5. Trích xuất đặc trưng

Mục tiêu: chuyển vùng ảnh trái cây thành vector số để mô hình học máy xử lý.

### 5.1. Đặc trưng màu sắc

Báo cáo nêu:

- Giá trị trung bình màu.
- Độ lệch chuẩn màu.
- Histogram màu.

Ý nghĩa:

- Trung bình: mô tả màu sắc tổng quát.
- Độ lệch chuẩn: phản ánh mức độ phân tán/đồng nhất.
- Histogram: mô tả phân bố màu.

Lưu ý: cần hạn chế ảnh hưởng của nền, bóng đổ và phản sáng; quy luật màu thay đổi khác nhau theo từng loại trái cây.

### 5.2. Đặc trưng kết cấu

Hai phương pháp được nêu:

- **GLCM (Gray-Level Co-occurrence Matrix):** thống kê quan hệ giữa các cặp mức xám; có thể tính contrast, homogeneity, energy và correlation.
- **LBP (Local Binary Patterns):** so sánh pixel trung tâm với các pixel lân cận, mã hóa mẫu nhị phân và dùng histogram để mô tả kết cấu cục bộ.

Các đặc trưng này chịu ảnh hưởng bởi ánh sáng, độ phân giải và tham số xử lý. Cần đánh giá hiệu quả trước khi kết hợp.

### 5.3. Đặc trưng hình dạng/kích thước

Tính từ đường biên hoặc mask trái cây, có thể gồm:

- Diện tích.
- Chu vi.
- Chiều rộng.
- Chiều cao.
- Độ tròn.

Độ tròn được mô tả theo công thức:

\[
Circularity = \frac{4\pi A}{P^2}
\]

Trong đó `A` là diện tích và `P` là chu vi. Giá trị càng gần 1 thì hình dạng càng gần hình tròn hoàn hảo.

Kích thước theo pixel phụ thuộc khoảng cách chụp và độ phân giải. Đặc trưng hình dạng không trực tiếp quyết định chất lượng, nên kết hợp với màu sắc và kết cấu.

### 5.4. Chuẩn hóa và PCA

Chương 2 dự kiến có:

- Chuẩn hóa đặc trưng.
- Giảm chiều bằng PCA.

Lưu ý triển khai:

- Chỉ `fit` scaler/PCA trên tập train.
- Dùng lại scaler/PCA đã fit cho validation, test và dữ liệu dự đoán mới.
- Tránh data leakage.
- Lưu scaler/PCA cùng model trong pipeline nếu cần.
- Không bắt buộc PCA nếu thực nghiệm cho thấy không có lợi; phải ghi nhận kết quả và lý do.

---

## 6. Các mô hình học máy cần xem xét

Báo cáo đề cập các thuật toán sau:

| Mô hình | Vai trò/đặc điểm theo báo cáo |
|---|---|
| KNN | Dự đoán dựa trên các mẫu gần nhất; nhạy với thang đo đặc trưng, cần cân nhắc chuẩn hóa |
| Logistic Regression | Phân loại dựa trên xác suất; nhanh, dễ diễn giải; hạn chế với quan hệ phi tuyến phức tạp |
| Naive Bayes | Dựa trên định lý Bayes và giả định độc lập có điều kiện giữa các đặc trưng |
| SVM | Tìm siêu phẳng phân chia các lớp; được sử dụng chủ yếu cho phân loại |
| Decision Tree | Dễ diễn giải, mô hình hóa quan hệ phi tuyến; cây sâu có nguy cơ overfitting |
| Random Forest | Kết hợp nhiều cây trên mẫu bootstrap và tập đặc trưng ngẫu nhiên |
| Gradient Boosting | Xây dựng tuần tự các mô hình, mô hình sau điều chỉnh lỗi của mô hình trước |

### 6.1. Yêu cầu so sánh

- Dùng cùng cách chia dữ liệu và quy trình đánh giá để so sánh công bằng.
- Tinh chỉnh siêu tham số bằng Grid Search theo kế hoạch của Chương 2.
- Ghi nhận cấu hình, tham số và thời gian chạy.
- Chọn mô hình tốt nhất dựa trên kết quả validation/cross-validation; chỉ đánh giá cuối cùng trên test sau khi đã lựa chọn.
- Không chọn mô hình chỉ dựa trên accuracy nếu dữ liệu mất cân bằng.

### 6.2. Baseline

Chương 2 yêu cầu có mô hình baseline để đối chứng nhưng chưa chỉ rõ baseline cụ thể. Agent cần đề xuất baseline đơn giản phù hợp với nhãn và dữ liệu hiện tại, ghi rõ lựa chọn và không trình bày đó là quyết định đã có trong báo cáo.

---

## 7. Huấn luyện và quản lý dữ liệu

### 7.1. Chia tập dữ liệu

Báo cáo yêu cầu có:

- Training set.
- Validation set.
- Test set.

Tỷ lệ chia cụ thể chưa được báo cáo quy định. Agent cần kiểm tra kích thước dataset, phân bố lớp và nhóm ảnh trước khi đề xuất tỷ lệ.

Yêu cầu:

- Giữ test set độc lập.
- Dùng stratified split khi phù hợp với bài toán phân loại.
- Nếu nhiều ảnh thuộc cùng một quả/lô/chủ thể, xem xét chia theo nhóm để tránh ảnh gần trùng xuất hiện ở nhiều tập.
- Không dùng test set để tinh chỉnh siêu tham số.
- Ghi seed và cách chia để tái lập kết quả.

### 7.2. Data augmentation

Báo cáo yêu cầu tăng cường dữ liệu **chỉ dành cho tập train**.

Agent cần bảo đảm augmentation không làm sai nhãn. Không áp dụng augmentation lên validation/test. Các phép biến đổi cụ thể phải phù hợp với ảnh trái cây và được ghi lại.

### 7.3. Mất cân bằng lớp và overfitting

Báo cáo yêu cầu xem xét:

- Xử lý mất cân bằng lớp.
- Hạn chế overfitting.
- Cross-validation.
- Regularization (L1/L2) khi phù hợp.
- Theo dõi chênh lệch hiệu năng train và validation.

Không áp dụng kỹ thuật chỉ để làm đẹp số liệu. Ghi nhận tác động thực nghiệm.

---

## 8. Đánh giá mô hình

### 8.1. Confusion Matrix

Dùng để so sánh nhãn thực tế và nhãn dự đoán.

Trong phân loại nhị phân:

- `TP`: dự đoán đúng mẫu dương tính.
- `TN`: dự đoán đúng mẫu âm tính.
- `FP`: mẫu âm tính bị dự đoán thành dương tính.
- `FN`: mẫu dương tính bị dự đoán thành âm tính.

Với phân loại đa lớp, đường chéo chính biểu thị dự đoán đúng; ngoài đường chéo là các trường hợp phân loại nhầm.

### 8.2. Các chỉ số

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

\[
Precision = \frac{TP}{TP + FP}
\]

\[
Recall = \frac{TP}{TP + FN}
\]

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

Với đa lớp, cần nêu rõ cách tổng hợp chỉ số (ví dụ macro, micro hoặc weighted) trong báo cáo kết quả.

### 8.3. ROC và AUC

- ROC biểu diễn TPR/Recall theo FPR ở các ngưỡng quyết định.
- AUC là diện tích dưới đường ROC.
- AUC gần 1 thể hiện khả năng phân biệt tốt hơn; AUC = 0.5 tương đương đoán ngẫu nhiên theo mô tả trong báo cáo.

Với bài toán đa lớp, cần nêu rõ chiến lược tính ROC/AUC (ví dụ one-vs-rest) và cách tổng hợp. Chỉ tạo ROC/AUC khi mô hình cung cấp điểm xác suất/decision score phù hợp.

### 8.4. Phân tích lỗi và diễn giải

Chương 2 và 3 yêu cầu:

- Feature importance/tầm quan trọng đặc trưng.
- Phân tích lỗi phân loại.
- Phân tích các trường hợp dự đoán đúng và sai.
- Learning curve và đánh giá overfitting.
- So sánh thời gian huấn luyện và dự đoán.

Chỉ báo cáo những kết quả thực sự được chạy và lưu lại.

---

## 9. Pipeline và chương trình

### 9.1. Pipeline dự kiến

Luồng tổng quát:

```text
Ảnh đầu vào
    |
    v
Kiểm tra ảnh / đọc ảnh
    |
    v
Tiền xử lý ảnh
    |
    v
Chuyển đổi không gian màu
    |
    v
Tách nền / phân đoạn vùng trái cây
    |
    v
Trích xuất đặc trưng màu sắc
    |
    +----> Trích xuất đặc trưng kết cấu (GLCM/LBP)
    |
    +----> Trích xuất đặc trưng hình dạng
    |
    v
Ghép vector đặc trưng
    |
    v
Chuẩn hóa / PCA (nếu được chọn)
    |
    v
Mô hình dự đoán
    |
    +----> Kết quả độ chín
    |
    +----> Kết quả chất lượng
    |
    v
Hiển thị kết quả
```

Đây là luồng định hướng từ nội dung báo cáo. Cách triển khai cụ thể phụ thuộc repository và cấu trúc nhãn thực tế.

### 9.2. Lưu và nạp mô hình

Chương 2 yêu cầu lưu và nạp mô hình đã huấn luyện. Khi triển khai:

- Lưu toàn bộ thành phần cần thiết để dự đoán nhất quán: model, scaler, PCA, cấu hình đặc trưng và mapping nhãn nếu có.
- Có kiểm tra tương thích giữa phiên bản pipeline và model.
- Không `fit` lại preprocessing khi dự đoán ảnh mới.
- Xử lý lỗi khi thiếu file model hoặc dữ liệu đầu vào không hợp lệ.
- Ghi lại cách huấn luyện và cách nạp model.

Định dạng lưu cụ thể chưa được báo cáo quy định; hãy dùng định dạng phù hợp với stack hiện tại.

### 9.3. Giao diện

Báo cáo yêu cầu giao diện có tối thiểu:

- Chọn/nhập ảnh.
- Hiển thị ảnh đầu vào.
- Hiển thị kết quả phân loại.

Không có quy định framework UI, phong cách thiết kế, API hay yêu cầu đăng nhập. Agent phải khảo sát dự án hiện tại và không tự ý thêm chức năng ngoài phạm vi nếu không cần thiết.

---

## 10. Cấu trúc báo cáo cần hỗ trợ

Agent cần tổ chức kết quả, biểu đồ, bảng và thông tin kỹ thuật để nhóm có thể hoàn thiện báo cáo theo bố cục sau.

### Chương 1. Cơ sở lý thuyết

1.1. Tổng quan về học máy  
- 1.1.1. Khái niệm học máy  
- 1.1.2. Quy trình xây dựng hệ thống học máy  
- 1.1.3. Học có giám sát và bài toán phân loại  

1.2. Bài toán phân loại ảnh và thị giác máy tính  
- 1.2.1. Khái niệm và các bước phân loại ảnh  
- 1.2.2. Phân loại nhị phân và đa lớp  
- 1.2.3. Thách thức về ánh sáng, nền ảnh, góc chụp  

1.3. Xử lý ảnh số cơ bản  
- 1.3.1. Biểu diễn ảnh và RGB/HSV/Lab  
- 1.3.2. Lọc nhiễu và cân bằng histogram  
- 1.3.3. Otsu và tách nền  

1.4. Trích xuất đặc trưng ảnh trái cây  
- 1.4.1. Đặc trưng màu sắc  
- 1.4.2. Đặc trưng kết cấu GLCM/LBP  
- 1.4.3. Đặc trưng hình dạng và kích thước  

1.5. Các thuật toán học máy phân loại  
- 1.5.1. KNN, Naive Bayes, Logistic Regression  
- 1.5.2. SVM  
- 1.5.3. Decision Tree, Random Forest, Gradient Boosting  

1.6. Cơ sở về độ chín và chất lượng trái cây  
- 1.6.1. Các giai đoạn độ chín  
- 1.6.2. Tiêu chí chất lượng và phân hạng trái cây  
- 1.6.3. Liên hệ giữa màu sắc/kết cấu với chất lượng và độ chín  

1.7. Ứng dụng học máy trong nhận diện trái cây  
- 1.7.1. Phân loại chủng loại trái cây  
- 1.7.2. Đánh giá độ chín  
- 1.7.3. Đánh giá chất lượng và phát hiện hư hỏng  

1.8. Các chỉ số đánh giá mô hình  
- 1.8.1. Confusion Matrix, Accuracy, Precision, Recall, F1  
- 1.8.2. ROC và AUC  
- 1.8.3. Cross-validation và overfitting  

### Chương 2. Mô hình / phương pháp

- 2.1. Tổng quan phương pháp đề xuất
  - Sơ đồ quy trình tổng thể
  - Đầu vào/đầu ra
  - Lý do lựa chọn phương pháp
  - Chiến lược xử lý đồng thời độ chín và chất lượng
- 2.2. Tiền xử lý và phân đoạn ảnh
  - Resize, chuẩn hóa, khử nhiễu
  - Chuyển đổi không gian màu
  - Tách nền/phân đoạn
  - Xây dựng vector đặc trưng
- 2.3. Trích xuất và lựa chọn đặc trưng
  - Màu sắc
  - Kết cấu và hình dạng
  - Chuẩn hóa và PCA
- 2.4. Sử dụng các mô hình phân loại
  - KNN và Logistic Regression
  - SVM
  - Random Forest và Gradient Boosting
  - Baseline
- 2.5. Huấn luyện và tinh chỉnh
  - Train/validation/test
  - Grid Search
  - Data augmentation chỉ trên train
  - Xử lý mất cân bằng và overfitting
- 2.6. Đánh giá và diễn giải
  - Chỉ số đánh giá
  - Feature importance
  - Phân tích lỗi
- 2.7. Đóng gói và triển khai
  - Pipeline dự đoán
  - Lưu/nạp model
  - Công cụ và thư viện (báo cáo nêu Python, OpenCV, scikit-learn)

### Chương 3. Kết quả thực nghiệm

- 3.1. Mô tả bài toán
  - Phát biểu bài toán
  - Đầu vào/đầu ra
  - Môi trường và cấu hình thực nghiệm
- 3.2. Dataset
  - Nguồn và phương pháp thu thập
  - Định nghĩa/gán nhãn
  - Làm sạch, chuẩn hóa, trích xuất đặc trưng
  - Augmentation trên train
  - Thống kê và chia tập
- 3.3. Kết quả huấn luyện
  - Cấu hình mô hình
  - Cross-validation và Grid Search
  - So sánh hiệu năng
  - Learning curve và overfitting
- 3.4. Kết quả kiểm thử
  - Confusion Matrix và chỉ số từng lớp
  - ROC/AUC
  - Các trường hợp đúng/sai
  - Đặc trưng ảnh hưởng nhiều nhất
- 3.5. Giao diện chương trình
  - Kiến trúc
  - Giao diện nhập ảnh và kết quả
  - Thử nghiệm ảnh thực tế
- 3.6. So sánh
  - Điều kiện so sánh
  - Độ chính xác giữa các mô hình
  - Thời gian huấn luyện/dự đoán
  - So sánh với phương pháp/nghiên cứu khác
- 3.7. Thảo luận
  - Ưu/nhược điểm
  - Hạn chế và hướng cải tiến

Cuối báo cáo: Kết luận và hướng phát triển, tài liệu tham khảo, phụ lục.

---

## 11. Phân công nhiệm vụ nhóm

### Trần Văn Tiến

- Xác định bài toán, mục tiêu, phạm vi và tiêu chí đánh giá.
- Tìm hiểu cơ sở lý thuyết.
- Đánh giá, so sánh mô hình bằng Confusion Matrix, ROC/AUC và lựa chọn mô hình tốt nhất.
- Quản lý tiến độ, điều phối công việc.
- Đóng gói pipeline và kiểm tra chất lượng báo cáo cuối cùng.
- Phối hợp huấn luyện và tinh chỉnh mô hình.

### Đặng Xuân Thảo

- Tìm hiểu cơ sở lý thuyết.
- Làm sạch, tiền xử lý, gán nhãn độ chín/chất lượng và chia train/validation/test.
- Xây dựng, kiểm thử chức năng dự đoán.
- Phân tích trường hợp phân loại đúng/sai.
- Tích hợp model với giao diện, kiểm thử và sửa lỗi.
- Phối hợp huấn luyện và tinh chỉnh mô hình.

### Lý Văn Quyến

- Tìm hiểu cơ sở lý thuyết.
- Thu thập dataset.
- Nghiên cứu xử lý ảnh, tách nền và phân đoạn vùng trái cây.
- Trích xuất đặc trưng màu sắc, kết cấu GLCM/LBP, hình dạng và PCA.
- Thiết kế giao diện nhập ảnh, hiển thị ảnh và kết quả.
- Phối hợp huấn luyện và tinh chỉnh mô hình.

---

## 12. Các thông tin còn thiếu — phải xác minh trong repository

Những điểm sau chưa được xác định đầy đủ trong báo cáo. Agent cần tìm trong mã nguồn, dataset, notebook hoặc hỏi người dùng nếu không tìm thấy.

- [ ] Đường dẫn dataset thực tế.
- [ ] Dataset gồm những loại trái cây nào.
- [ ] Số lượng ảnh và phân bố ảnh theo lớp.
- [ ] Cấu trúc thư mục và định dạng annotation/label.
- [ ] Nhãn độ chín thực tế.
- [ ] Nhãn chất lượng thực tế.
- [ ] Chất lượng và độ chín được gán nhãn riêng hay kết hợp.
- [ ] Cách xử lý ảnh có nhiều trái cây trong cùng khung hình.
- [ ] Kích thước ảnh đầu vào.
- [ ] Phương pháp tách nền thực sự sử dụng.
- [ ] Danh sách đặc trưng cuối cùng và tham số trích xuất.
- [ ] Có sử dụng PCA hay không.
- [ ] Tỷ lệ train/validation/test.
- [ ] Baseline được chọn.
- [ ] Siêu tham số và phạm vi Grid Search.
- [ ] Mô hình tốt nhất sau thực nghiệm.
- [ ] Framework giao diện và cách tích hợp pipeline.
- [ ] Cấu hình máy và phiên bản thư viện.
- [ ] Lệnh cài đặt, huấn luyện, đánh giá và chạy ứng dụng.

Không được đánh dấu hoàn thành các mục này nếu chưa có bằng chứng.

---

## 13. Tiêu chí nghiệm thu kỹ thuật

### Dữ liệu

- [ ] Đọc được dataset theo cấu trúc thực tế.
- [ ] Kiểm tra ảnh lỗi, file trùng và nhãn không hợp lệ.
- [ ] Thống kê số lượng mẫu theo từng lớp.
- [ ] Chia dữ liệu có thể tái lập.
- [ ] Không có data leakage giữa train/validation/test.

### Xử lý và đặc trưng

- [ ] Pipeline xử lý ảnh chạy được trên ảnh đầu vào.
- [ ] Có kiểm tra kết quả tách nền/mask.
- [ ] Vector đặc trưng có số chiều và thứ tự ổn định.
- [ ] Có xử lý trường hợp ảnh không đọc được hoặc không tách được đối tượng.
- [ ] Preprocessing lúc dự đoán giống lúc huấn luyện.

### Huấn luyện và đánh giá

- [ ] Có baseline.
- [ ] Huấn luyện được các mô hình khả thi trong phạm vi đề tài.
- [ ] Lưu cấu hình và tham số.
- [ ] Có kết quả validation/cross-validation.
- [ ] Đánh giá test độc lập.
- [ ] Có confusion matrix và các chỉ số phù hợp.
- [ ] Có ROC/AUC nếu phù hợp với mô hình và nhãn.
- [ ] Có phân tích lỗi và so sánh mô hình.
- [ ] Không ghi số liệu giả.

### Ứng dụng

- [ ] Có thể chọn ảnh đầu vào.
- [ ] Hiển thị ảnh đã chọn.
- [ ] Hiển thị kết quả độ chín/chất lượng theo nhãn model hỗ trợ.
- [ ] Có thông báo lỗi dễ hiểu khi ảnh không hợp lệ hoặc model chưa sẵn sàng.
- [ ] Có hướng dẫn cài đặt và chạy ứng dụng.

### Tài liệu

- [ ] README có hướng dẫn cài đặt và chạy.
- [ ] Có mô tả dataset và nhãn.
- [ ] Có hướng dẫn huấn luyện/đánh giá.
- [ ] Có mô tả pipeline và model đã lưu.
- [ ] Có bảng/biểu đồ kết quả thực nghiệm thực tế.
- [ ] Có ghi chú hạn chế và phần chưa triển khai.

---

## 14. Các vấn đề cần lưu ý trong tài liệu nguồn

Báo cáo Word có một số chỗ chưa hoàn chỉnh hoặc chưa nhất quán. Coding agent không được tự động coi các nội dung này là yêu cầu triển khai:

1. Chương 2 và Chương 3 chủ yếu mới có tiêu đề/mục lục, chưa có mô tả thực nghiệm chi tiết.
2. Phần kết luận ở cuối file có nội dung nhắc **Orange Data Mining** và bài toán **dự đoán mức độ ung thư phổi**, không phù hợp với đề tài phân loại chất lượng và độ chín trái cây. Không dùng nội dung này làm đặc tả dự án; cần được nhóm chỉnh sửa khi hoàn thiện báo cáo.
3. Danh sách bảng, danh sách hình và danh sách từ viết tắt còn trống.
4. Một số công thức/biểu thức trong file Word bị mất ký hiệu hoặc hiển thị không đầy đủ khi trích xuất.
5. Phần phân loại chất lượng nêu tiêu chí thương mại, nhưng chưa chứng minh dataset đang dùng có các nhãn Extra Class, Class 1, Class 2 và hạng chế biến.
6. Phần lý thuyết có đề cập CNN và Vision Transformer, nhưng phần phương pháp dự kiến tập trung vào các mô hình học máy truyền thống dựa trên đặc trưng.

---

## 15. Định dạng báo cáo tiến độ của coding agent

Sau mỗi giai đoạn, báo cáo theo mẫu:

```text
## Tóm tắt
- Đã thực hiện:
- Kết quả:

## File đã thay đổi
- path/to/file — mục đích thay đổi

## Kiểm thử
- Lệnh:
- Kết quả:
- Kiểm thử chưa chạy (nếu có):

## Vấn đề còn tồn tại
- ...

## Giả định / quyết định cần xác nhận
- ...
```

Khi hoàn thành một tính năng, cập nhật checklist tại tài liệu này hoặc tài liệu tiến độ riêng trong `docs/`.

---

## 16. Chỉ dẫn bắt đầu cho agent

**Bây giờ hãy khảo sát repository hiện tại trước khi viết hoặc sửa code.**

1. Xác nhận thư mục gốc dự án và trạng thái Git.
2. Liệt kê cấu trúc thư mục ở mức cần thiết; không dump toàn bộ thư mục dữ liệu lớn.
3. Đọc tất cả tài liệu liên quan trong `docs/`.
4. Xác định dataset, nhãn, pipeline, model và giao diện hiện có.
5. Lập bảng đối chiếu:
   - Đã có.
   - Còn thiếu.
   - Có nhưng chưa đúng/chưa kiểm thử.
   - Chưa thể xác định.
6. Đề xuất thứ tự triển khai ưu tiên, nêu rõ file dự kiến sẽ thay đổi.
7. **Chưa refactor lớn, chưa xóa dữ liệu và chưa thay đổi công nghệ** trước khi hoàn tất khảo sát.
8. Sau khi khảo sát, trình bày kết quả và bắt đầu triển khai các phần rõ ràng, an toàn, có thể kiểm thử.

Mục tiêu cuối cùng là một chương trình phân loại độ chín và chất lượng trái cây có pipeline nhất quán, kết quả thực nghiệm có thể tái lập, giao diện sử dụng được và tài liệu đủ để nhóm cài đặt, chạy thử và trình bày.
