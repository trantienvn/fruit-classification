# Lý thuyết và Đánh giá Rủi ro: Trích xuất Đặc trưng bằng CNN

> **Trạng thái tài liệu:** Tài liệu phân tích lý thuyết và đánh giá rủi ro tham khảo. Nội dung dưới đây **chưa thay đổi mã nguồn hay pipeline huấn luyện hiện hành** của dự án.

---

## 1. Hiện trạng kỹ thuật đã xác nhận trong repository

Để đánh giá đúng bối cảnh khi cân nhắc đưa Mạng nơ-ron tích chập (CNN) vào khâu trích xuất đặc trưng, cần đối chiếu với trạng thái thực tế của mã nguồn và tài liệu:

- **Biểu diễn đầu vào hiện hành ([`src/preprocess.py`](../src/preprocess.py)):** Ảnh RGB được resize về `64×64`, chuẩn hóa miền `[0.0, 1.0]` kiểu `float32` và làm phẳng thành vector số `12.288` chiều (`representation = "pixels_rgb_0_1"`), lưu cache trong `data/processed/numeric/numeric_dataset.npz`.
- **Mô-đun đặc trưng thủ công tham chiếu ([`src/features.py`](../src/features.py)):** Chứa bộ trích xuất `90` đặc trưng (70 màu sắc RGB/HSV/Lab + 14 kết cấu LBP/GLCM + 6 hình dạng dựa trên mask tách nền Otsu); hiện không được [`src/train.py`](../src/train.py) và [`src/predict.py`](../src/predict.py) gọi trong luồng mặc định.
- **Pipeline huấn luyện ([`src/train.py`](../src/train.py)):** Nạp ma trận số từ `.npz`, áp dụng `StandardScaler` + `PCA` (mặc định giữ 95% phương sai), huấn luyện và so sánh `9` estimator `scikit-learn` trên CPU (`Dummy`, `KNN`, `NaiveBayes`, `LogisticRegression`, `SVM-RBF`, `SVM-Linear`, `DecisionTree`, `RandomForest`, `GradientBoosting`), tinh chỉnh top-3 bằng `GridSearchCV` 5-fold, sau đó refit với ảnh tăng cường (`augment_training_images`) và huấn luyện thêm bộ mô hình `per-fruit`.
- **Dataset ([`data/README.md`](../data/README.md)):**
  - Dataset mặc định `data/processed/dataset/` (9 lớp `Orange`/`Banana`/`Mango` × `Unripe`/`Ripe`/`Spoiled`): `1.260` ảnh train (`140` ảnh/lớp), `271` validation, `270` test (`14` kích thước ảnh khác nhau).
  - Dataset tùy chọn `data/processed/whole-scene/` (từ `data/raw/`, 9 lớp `Apple`/`Banana`/`Mango` × `Unripe`/`Ripe`/`Overripe`): `1.547` ảnh train, `443` validation, `220` test.
- **Ràng buộc môi trường và phạm vi ([`AGENTS.md`](../AGENTS.md), [`requirements.txt`](../requirements.txt), [`docs/AGENT_PROJECT_SPEC.md`](AGENT_PROJECT_SPEC.md)):**
  - Chuẩn hóa **Python 3.14**, chạy CPU với `scikit-learn`, `Pillow`, `OpenCV`, `scikit-image`, `FastAPI`; hiện **không cài `torch`, `torchvision` hay `onnxruntime`**.
  - Không tự cài thư viện nặng hay tự tải trọng số pretrained ngoài bước người dùng chủ động huấn luyện; bộ test phải chạy độc lập với dataset và không yêu cầu tải mô hình.
  - Đặc tả học phần định hướng ML truyền thống trên CPU; mọi chuyển đổi hướng tiếp cận sang Deep Learning/CNN cần có sự xác nhận rõ ràng.

---

## 2. Đánh giá rủi ro ngắn gọn

| Nhóm rủi ro | Mức độ | Phân tích cụ thể trong dự án | Hướng giảm nhẹ |
|---|---|---|---|
| **Lệch phạm vi học phần** | Cao | [AGENT_PROJECT_SPEC.md](AGENT_PROJECT_SPEC.md) và [context/decisions.md](../context/decisions.md) định hướng ML truyền thống (`KNN`, `SVM`, `RF`, `LR`...) trên CPU; thay bằng CNN end-to-end dễ sai định hướng báo cáo. | Chỉ dùng CNN làm **bộ trích xuất đặc trưng cố định (frozen backbone)** để tạo vector đầu vào cho `9` classifier `scikit-learn` hiện có. |
| **Môi trường Python 3.14 & thư viện nặng** | Cao | [requirements.txt](../requirements.txt) không có framework DL; cài `torch`/`torchvision` làm phình môi trường, có thể vướng tương thích wheel trên Python 3.14 và vi phạm ràng buộc giữ nhẹ của [AGENTS.md](../AGENTS.md). | Nếu triển khai inference-only, ưu tiên backbone nhẹ chạy qua `onnxruntime` trên CPU thay vì kéo cả stack training CUDA; chỉ tải weights khi user chạy lệnh. |
| **Dữ liệu nhỏ & lệch miền ImageNet** | Trung bình–Cao | Tập train mặc định chỉ có `1.260` ảnh (`140` ảnh/lớp): huấn luyện CNN từ đầu hoặc fine-tune sâu rất dễ overfit. Pretrained ImageNet tối ưu cho *nhận dạng loại vật thể* (bất biến màu phần nào) hơn là *phân biệt độ chín/đốm hỏng nhỏ* (`Unripe`/`Ripe`/`Spoiled`). | Không train từ đầu; dùng frozen pretrained kết hợp đối chứng trên `val`, cân nhắc ghép thêm đặc trưng tầng trung hoặc chỉ số màu HSV/Lab từ [`src/features.py`](../src/features.py). |
| **Đứt gãy pipeline, Web & kiểm thử** | Trung bình | [`FruitClassifier.load`](../src/predict.py) khóa cứng `feature_type == "pixels_rgb_0_1"` và `n_features`; [`_plot_feature_importance`](../src/evaluate.py) giảm tính diễn giải vì chiều embedding trừu tượng; độ trễ API web (`app/web/`) tăng trên CPU; test offline dễ hỏng nếu gọi tải weights thật. | Phân nhánh rõ theo `feature_type` trong `config.json`/`.npz`, đo latency thực tế cho Web API, và dùng mock extractor trong `tests/`. |
| **Rò rỉ dữ liệu & lệch chuẩn hóa** | Trung bình | Augmentation hiện chỉ chạy lúc refit cuối từ vector đã cache; đưa CNN vào bước tiền xử lý dễ làm sai lệch chuẩn hóa kênh ảnh (ImageNet `mean`/`std`) giữa lúc train và inference. | Dùng chung một hàm biến đổi tiền xử lý ảnh cho cả `preprocess` và `FruitClassifier`; giữ nguyên khử trùng lặp `test > val > train` ([`deduplicate_splits`](../src/dataset.py)). |

---

## 3. Cơ sở lý thuyết: CNN trong trích xuất đặc trưng cho dự án

### 3.1. Động cơ: Hạn chế của biểu diễn hiện tại và vai trò của CNN

Trong dự án hiện có hai cách biểu diễn ảnh truyền thống:

1. **Vector pixel thô `64×64×3` (`12.288` chiều, [`src/preprocess.py`](../src/preprocess.py)):**
   Coi mỗi tọa độ `(y, x, c)` là một biến độc lập sau khi làm phẳng. Biểu diễn này phá vỡ cấu trúc lân cận 2D, số chiều rất lớn so với `1.260` mẫu train và cực kỳ nhạy với việc quả dịch chuyển vị trí, xoay góc nhẹ, thay đổi nền hoặc kích thước khung hình.
2. **Vector `90` đặc trưng thủ công ([`src/features.py`](../src/features.py)):**
   Cung cấp các chỉ số có ý nghĩa vật lý rõ ràng (histogram RGB/HSV, thống kê Lab, LBP/GLCM, hình dạng), nhưng phụ thuộc trực tiếp vào bước tách nền [`segment_fruit`](../src/features.py) bằng ngưỡng Otsu. Với ảnh nền phức tạp, bóng đổ mạnh hoặc màu quả tiệp màu nền (đặc biệt trên tập `whole-scene`), mask tách nền sai sẽ làm nhiễu cả đặc trưng màu sắc lẫn hình dạng.

**Mạng nơ-ron tích chập (CNN)** giải quyết đồng thời hai vấn đề này bằng cơ chế tự động học các bộ lọc không gian phân cấp trực tiếp từ lưới pixel 2D mà không cần bước tách nền nhị phân thủ công.

### 3.2. Cơ chế toán học cốt lõi của CNN Feature Extractor

Khi đóng vai trò **bộ trích xuất đặc trưng (backbone)**, CNN loại bỏ tầng phân loại tuyến tính cuối cùng (`Linear`/`Softmax`) và biến ảnh RGB đầu vào $I \in \mathbb{R}^{H \times W \times 3}$ thành một vector đặc trưng cô đọng $z \in \mathbb{R}^d$ thông qua ba khối xử lý:

1. **Tầng tích chập (Convolutional Layer) — Kết nối cục bộ và chia sẻ trọng số:**
   Mỗi bộ lọc (kernel) $K_k$ trượt qua toàn bộ không gian 2D của ảnh hoặc bản đồ đặc trưng trước đó:
   $$(I * K_k)(i, j) = \sum_{c=1}^{3} \sum_{m=-r}^{r} \sum_{n=-r}^{r} I_c(i+m, j+n)\, K_{k,c}(m, n) + b_k$$
   Nhờ **chia sẻ trọng số** trên toàn ảnh, một bộ lọc phát hiện cạnh cong hoặc đốm nâu hoạt động đồng nhất dù trái cây nằm ở bất kỳ vị trí nào trong khung hình (*tính đồng biến dịch chuyển — translation equivariance*), khắc phục nhược điểm cố định tọa độ của vector pixel làm phẳng.

2. **Hàm kích hoạt phi tuyến và Gộp vùng (Pooling / Strided Convolution):**
   - Các hàm kích hoạt phi tuyến (`ReLU`, `Hardswish`, `SiLU`) cho phép mạng biểu diễn tổ hợp phức tạp giữa sắc độ và kết cấu bề mặt.
   - Các phép giảm mẫu không gian (Max/Average Pooling hoặc tích chập có `stride > 1`) mở rộng dần **vùng cảm thụ (receptive field)** của mỗi neuron ở các tầng sâu hơn, giúp biểu diễn ổn định hơn trước các dịch chuyển nhỏ và nhiễu nền.

3. **Global Average Pooling (GAP) — Tạo vector embedding cố định:**
   Tại cuối phần thân mạng, tensor đặc trưng có kích thước $F \in \mathbb{R}^{H' \times W' \times C}$ (với $C$ là số kênh đặc trưng). Phép GAP lấy trung bình theo hai chiều không gian trên từng kênh $k \in \{1, \dots, C\}$:
   $$z_k = \frac{1}{H' \cdot W'} \sum_{i=1}^{H'} \sum_{j=1}^{W'} F_k(i, j)$$
   Đầu ra là vector $z = [z_1, z_2, \dots, z_C]^\top \in \mathbb{R}^C$ (ví dụ `576` chiều với MobileNetV3-Small hoặc `1.280` chiều với MobileNetV2/EfficientNet-B0) — gọn hơn nhiều lần so với `12.288` chiều pixel thô.

### 3.3. Ý nghĩa phân cấp đặc trưng đối với bài toán 9 lớp trái cây

Bài toán của dự án yêu cầu phân biệt đồng thời **chủng loại quả** (`Orange`/`Banana`/`Mango` hoặc `Apple`/`Banana`/`Mango`) và **trạng thái độ chín / hư hỏng** (`Unripe`, `Ripe`, `Spoiled`/`Overripe`). Các tầng CNN phản ánh thông tin theo 3 cấp độ:

- **Các tầng nông (Low-level):** Bắt cạnh biên, độ tương phản cục bộ, chuyển tiếp màu sắc (xanh lục, vàng, cam, nâu đen) và vi kết cấu vỏ (vỏ sần đặc trưng của cam, vết trầy hoặc đốm nhỏ ban đầu).
- **Các tầng trung (Mid-level):** Tổ hợp thành các họa tiết vùng bề mặt — giúp phân biệt đốm nâu lấm tấm tự nhiên trên chuối chín với vùng thâm nhũn, mốc hoặc thối lan rộng ở lớp `Spoiled`, cũng như sự chuyển màu loang trên vỏ xoài.
- **Các tầng sâu (High-level):** Mã hóa cấu trúc hình thái tổng thể của từng loại quả (dáng thuôn cong của chuối, dáng tròn của cam, dáng bầu dục của xoài) và tách biệt đối tượng chính khỏi nền xung quanh tốt hơn ngưỡng Otsu.

### 3.4. Kiến trúc lai: Transfer Learning đóng băng kết hợp ML truyền thống

Với quy mô dữ liệu huấn luyện hiện có (`1.260` ảnh ở bộ thư mục, `1.547` ảnh ở `whole-scene`), việc huấn luyện CNN từ đầu (from scratch) dễ dẫn đến **overfitting** nghiêm trọng do số lượng tham số mạng lớn hơn nhiều so với số mẫu dữ liệu.

Do đó, chiến lược phù hợp nhất với cả lý thuyết học máy trên tập dữ liệu nhỏ lẫn kiến trúc CPU hiện tại của dự án là **Hybrid Transfer Learning (Frozen CNN Feature Extraction + Traditional ML Classifier)**:

```text
Ảnh RGB (resize & chuẩn hóa theo backbone, vd 224×224, mean/std ImageNet)
   │
   ▼  [Frozen Pretrained CNN Backbone — bỏ lớp Softmax cuối, lấy vector GAP]
Vector embedding z ∈ R^d   (thay cho 12.288 pixel thô trong .npz)
   │
   ▼  [Pipeline scikit-learn hiện có trong src/train.py]
StandardScaler ──▶ (Tùy chọn) PCA ──▶ 9 Classifier (SVM-RBF, LogisticRegression, RF, KNN, ...)
```

Cách tiếp cận này có 3 ưu điểm lý thuyết rõ rệt:
1. Tận dụng tri thức thị giác phổ quát (biên, kết cấu, hình khối) đã học trên bộ dữ liệu lớn (như ImageNet) mà không cần lan truyền ngược (backpropagation) qua hàng triệu tham số trên tập ảnh nhỏ.
2. Không yêu cầu GPU trong quá trình huấn luyện bộ phân loại; chỉ cần trích xuất embedding một lần để lưu cache `.npz` (giữ nguyên triết lý lưu cache của [`src/preprocess.py`](../src/preprocess.py)).
3. Duy trì nguyên vẹn toàn bộ khung so sánh `9` thuật toán, `GridSearchCV` 5-fold, cơ chế `per-fruit` và báo cáo đánh giá của [`src/train.py`](../src/train.py) và [`src/evaluate.py`](../src/evaluate.py).

### 3.5. Tương tác với `StandardScaler`, `PCA` và 9 thuật toán phân loại

Khi thay vector pixel thô `12.288` chiều bằng vector CNN embedding $z \in \mathbb{R}^d$:

- **Với `StandardScaler` ([`_preprocess`](../src/train.py)):** Các chiều kích hoạt sau tầng phi tuyến (`ReLU`/`Hardswish`) thường có phân bố lệch và biên độ phương sai rất khác nhau giữa các kênh. Chuẩn hóa về trung bình `0`, phương sai `1` vẫn cần thiết cho `SVM-RBF`, `SVM-Linear`, `LogisticRegression` và `KNN`. Ngoài ra, chuẩn hóa chuẩn $L_2$ ($\hat{z} = z / \|z\|_2$) thường giúp các phép đo góc/khoảng cách ổn định hơn.
- **Với `PCA(0.95)`:** Khác với pixel thô (chứa rất nhiều nhiễu không gian cần nén mạnh), embedding CNN ở tầng GAP đã là biểu diễn nén ngữ nghĩa. Việc cắt bỏ `5%` phương sai đuôi bằng PCA có thể vô tình loại bỏ tín hiệu nhỏ liên quan đến đốm hỏng (`Spoiled`) hoặc ranh giới chín sớm (`Unripe`/`Ripe`). Cần chạy đối chứng `--pca` và `--no-pca` trên tập `val` thay vì mặc định bật PCA.
- **Tác động dự kiến lên các nhóm mô hình trong `MODELS`:**
  - **Nhóm tuyến tính & Kernel (`LogisticRegression`, `SVM-Linear`, `SVM-RBF`, `KNN`):** Hưởng lợi rõ nhất vì không gian embedding tầng cuối của CNN vốn được thiết kế gần **tách tuyến tính (linearly separable)**; `SVM-RBF` và `LogisticRegression` thường khai thác tốt ranh giới phân tách mượt giữa các mức độ chín.
  - **Nhóm Bayes & Cây (`GaussianNB`, `DecisionTree`, `RandomForest`, `GradientBoosting`):** `GaussianNB` dễ bị ảnh hưởng do các chiều embedding CNN có tương quan chặt (vi phạm giả định độc lập có điều kiện); các mô hình cây chia trực giao theo từng trục đơn lẻ thường kém ưu thế hơn mô hình tuyến tính/kernel trên biểu diễn liên tục dày đặc với cỡ mẫu nhỏ.

### 3.6. Lưu ý lý thuyết riêng cho phân loại độ chín & hư hỏng

Một điểm lý thuyết quan trọng khi dùng pretrained CNN từ ImageNet là **mục tiêu tiền huấn luyện gốc thiên về phân loại danh mục vật thể** (nhiều lớp có xu hướng bất biến màu sắc nhất định, ví dụ táo xanh và táo đỏ đều cùng nhãn "apple"), trong khi bài toán của dự án lại cần phân biệt tinh tế giữa `Unripe`, `Ripe` và `Spoiled`/`Overripe`.

Hai cơ chế cải thiện về mặt lý thuyết nếu trích xuất tầng cuối chưa đủ phân biệt độ chín:
1. **Kết hợp đa tầng (Multi-layer Feature Fusion):** Trích xuất thêm thống kê trung bình + độ lệch chuẩn theo không gian từ một block tầng trung gian (nơi thông tin màu sắc và đốm vỏ cục bộ chưa bị nén hoàn toàn) ghép với vector GAP tầng cuối.
2. **Ghép lai CNN + Đặc trưng màu (Hybrid CNN + Color):** Ghép vector embedding CNN với các thống kê màu toàn cục (HSV/Lab, không phụ thuộc tách nền Otsu dễ lỗi) để bảo toàn tín hiệu sắc độ tuyệt đối.

---

## 4. Lựa chọn Backbone và Runtime phù hợp cho dự án CPU / Python 3.14

Dựa trên yêu cầu chạy CPU, kích thước dataset (`1.260` ảnh train) và độ trễ API web ([`predict_upload`](../app/web/services/prediction_service.py)):

| Backbone tham khảo | Tham số (xấp xỉ) | Chiều embedding | Đánh giá lý thuyết cho dự án |
|---|---:|---:|---|
| **MobileNetV3-Small** | ~2.5M | 576 | Nhẹ nhất, suy luận CPU nhanh, tỷ lệ chiều/số mẫu train (`576` so với `1.260`) cân đối, phù hợp Web API CPU. |
| **MobileNetV2 / EfficientNet-B0** | ~3.5M – 5.3M | 1.280 | Giữ chi tiết kết cấu vỏ và đốm hỏng (`Spoiled`) tốt hơn ở kích thước `224×224`, nhưng tốn thời gian CPU hơn. |
| **ResNet-18** | ~11.7M | 512 | Dễ tách và ghép đặc trưng tầng trung (`layer3`) + tầng cuối (`layer4`), nhưng dung lượng và độ trễ CPU cao hơn nhóm MobileNet. |

Về **môi trường thực thi**: nếu chỉ sử dụng CNN ở chế độ đóng băng (inference-only, không fine-tune), định dạng **ONNX chạy qua `onnxruntime` trên CPU** giúp giảm đáng kể dung lượng phụ thuộc và rủi ro tương thích wheel trên Python 3.14 so với cài đầy đủ `torch` + `torchvision`.

---

## 5. So sánh tổng hợp 3 hướng biểu diễn đặc trưng

| Tiêu chí | Pixel thô `64×64×3` (Hiện hành, [`src/preprocess.py`](../src/preprocess.py)) | Thủ công `90` chiều (Tham chiếu, [`src/features.py`](../src/features.py)) | CNN Frozen Embedding (Phân tích đề xuất) |
|---|---|---|---|
| **Số chiều vector** | `12.288` | `90` | Thường `512 – 1.280` |
| **Tính bất biến vị trí / nền** | Kém (nhạy dịch chuyển và nền) | Trung bình (phụ thuộc tách nền Otsu) | Tốt (nhờ tích chập và vùng cảm thụ sâu) |
| **Độ nhạy sắc độ chín/hỏng** | Giữ pixel gốc nhưng lẫn nền | Rất rõ (histogram/thống kê HSV, Lab) | Tốt ở tầng nông/trung; tầng sâu có thể giảm nhạy màu |
| **Khả năng diễn giải** | Thấp (theo tọa độ pixel / PC) | Cao (biết rõ màu sắc, LBP, độ tròn) | Thấp (các chiều ẩn trừu tượng) |
| **Phụ thuộc & chi phí CPU** | Rất nhẹ, thuần NumPy/PIL | Nhẹ (OpenCV + scikit-image) | Cần runtime DL/ONNX + trọng số pretrained |

---

## 6. Bản đồ tác động kỹ thuật nếu triển khai vào codebase

Nếu quyết định tích hợp CNN vào mã nguồn ở bước sau, cần xử lý `4` điểm chạm kiến trúc để tránh phá vỡ hợp đồng dữ liệu và bộ kiểm thử hiện hành:

1. **Vấn đề augmentation không đảo ngược được ([`src/train.py`](../src/train.py)):**
   Pipeline hiện tại đọc `X_train` từ `.npz`, khôi phục lại ảnh bằng [`numeric_vectors_to_images`](../src/preprocess.py), tăng cường xoay/lật/đổi sáng rồi mới mã hóa lại thành vector. Nếu `X_train` là embedding CNN 1D thì **không thể tái tạo lại ảnh**. Khi dùng CNN, augmentation phải thực hiện trên ảnh gốc trước bước trích xuất đặc trưng (hoặc lưu cache riêng phần train tăng cường), đảm bảo `val`, `test` và các fold CV không nhận ảnh tăng cường.
2. **Định danh phiên bản đặc trưng & tương thích inference ([`src/preprocess.py`](../src/preprocess.py), [`src/predict.py`](../src/predict.py)):**
   [`FruitClassifier.load`](../src/predict.py) hiện từ chối mọi `config.json` không có `feature_type == "pixels_rgb_0_1"`. Cần phân nhánh rõ theo `feature_type` (giữ nguyên cấu hình cũ để tương thích ngược, đồng thời lưu đủ tên backbone, kích thước đầu vào, giá trị chuẩn hóa `mean`/`std` khi dùng CNN).
3. **Diễn giải kết quả trong báo cáo ([`src/evaluate.py`](../src/evaluate.py)):**
   Biểu đồ `feature_importance.png` tính trên embedding CNN (`cnn_feat_0`, `cnn_feat_1`...) không còn giải thích trực tiếp được yếu tố màu hay kết cấu nào quyết định nhãn; cần ghi nhận rõ giới hạn này trong phần thảo luận báo cáo (hoặc bổ sung phân tích trực quan bằng bản đồ nhiệt như Grad-CAM nếu phân tích sâu về CNN).
4. **Tính độc lập của bộ kiểm thử ([`tests/test_preprocess.py`](../tests/test_preprocess.py), [`tests/test_pipeline.py`](../tests/test_pipeline.py)):**
   Bộ test hiện tại khóa chặt mặc định `pixels_rgb_0_1` và chạy hoàn toàn offline. Mọi bài test cho nhánh CNN bắt buộc dùng mock/stub extractor trả về vector cố định, không tải trọng số hay yêu cầu dataset thật theo đúng nguyên tắc trong [`AGENTS.md`](../AGENTS.md).

---

## 7. Các điểm cần chốt trước khi chuyển sang triển khai

1. **Phạm vi học phần:** Dùng song song để đối chứng thực nghiệm (`pixels_rgb_0_1` vs `handcrafted` vs `CNN embedding`), hay giữ nguyên mã nguồn hiện tại và chỉ dùng tài liệu này cho phần thảo luận/hướng phát triển của báo cáo.
2. **Công nghệ runtime:** Chốt giữa `onnxruntime` (nhẹ cho inference CPU trên Python 3.14) hoặc `torch` + `torchvision` nếu nhóm chấp nhận phụ thuộc nặng hơn.
3. **Chiến lược đặc trưng độ chín:** Dùng embedding thuần của một backbone nhẹ (ví dụ MobileNetV3-Small) hay kết hợp với đặc trưng màu HSV/Lab để tối ưu phân biệt `Unripe`/`Ripe`/`Spoiled`.
