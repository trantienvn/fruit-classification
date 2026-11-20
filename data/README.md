# Dữ liệu

## 1. Dataset classification chính: Cam, Chuối, Xoài (9 lớp)

Dataset phân loại chính của dự án được lưu tại `data/processed/dataset/{train,val,test}/`.
Dataset gồm 9 lớp phân loại kết hợp giữa 3 loại quả (**Cam**, **Chuối**, **Xoài**) và 3 trạng thái độ chín/chất lượng (**Chưa chín / Xanh**, **Chín**, **Bị hỏng**). Các thư mục `*-hong` được xác nhận là trái cây hỏng (nấm mốc, thâm úng), không gộp vào nhãn quá chín.

Đây là tập dữ liệu chính thức phục vụ toàn bộ pipeline Học Sâu (`src_dl/`) và suy luận trực tiếp trên trình duyệt Web qua ONNX Runtime Web.

### Bảng phân bố số lượng ảnh theo lớp và tập dữ liệu

Mỗi lớp có đúng **140** ảnh train, **30** ảnh validation và **30** ảnh test (mỗi lớp đạt chuẩn **200** ảnh, cân bằng 100% giữa các lớp):

| Thư mục lớp | Train | Validation | Test | Tổng cộng |
|---|---:|---:|---:|---:|
| `cam-chin` | 140 | 30 | 30 | 200 |
| `cam-chua-chin` | 140 | 30 | 30 | 200 |
| `cam-hong` | 140 | 30 | 30 | 200 |
| `chuoi-chin` | 140 | 30 | 30 | 200 |
| `chuoi-chua-chin` | 140 | 30 | 30 | 200 |
| `chuoi-hong` | 140 | 30 | 30 | 200 |
| `xoai-chin` | 140 | 30 | 30 | 200 |
| `xoai-chua-chin` | 140 | 30 | 30 | 200 |
| `xoai-hong` | 140 | 30 | 30 | 200 |
| **Tổng cộng** | **1.260** | **270** | **270** | **1.800** |

### Kết quả kiểm toán dữ liệu (`python -m src.audit`)

- **Tổng số ảnh:** 1.800 ảnh; 100% giải mã thành công (0 ảnh lỗi / corrupt).
- **Tỷ lệ phân hoạch:** Đạt chuẩn cân bằng 70% Train (1.260 ảnh), 15% Validation (270 ảnh), 15% Test (270 ảnh).
- **Độ phân giải:** Có 13 kích thước ảnh khác nhau, phổ biến nhất là `224×224` (570 ảnh), `416×416` (287 ảnh), `512×512` (240 ảnh), `300×300` (189 ảnh), `255×255` (156 ảnh), `320×258` (125 ảnh), `640×480` (107 ảnh).
- **Trùng lặp:** Phát hiện 2 nhóm trùng byte chính xác (1 nhóm nằm trong `train/cam-chin`, 1 nhóm cross-split giữa `train/cam-hong` và `val/cam-hong`). Pipeline loại bản sao khỏi split ưu tiên thấp hơn khi trích xuất đặc trưng nếu cần, giữ thứ tự ưu tiên `test > val > train`.

### Ánh xạ tên thư mục sang nhãn mô hình

| Nhãn mô hình tiếng Anh | Nhãn tiếng Việt | Thư mục dữ liệu | Phẩm cấp chất lượng |
|---|---|---|---|
| Orange Unripe | Cam · Chưa chín | `cam-chua-chin` | Đạt chuẩn tiêu thụ (Tốt - Loại 2) |
| Orange Ripe | Cam · Chín | `cam-chin` | Đạt chuẩn (Tốt - Loại 1) |
| Orange Spoiled | Cam · Bị hỏng | `cam-hong` | Không đạt chuẩn (Kém - Loại bỏ) |
| Banana Unripe | Chuối · Chưa chín | `chuoi-chua-chin` | Đạt chuẩn lưu kho (Tốt - Loại 2) |
| Banana Ripe | Chuối · Chín | `chuoi-chin` | Đạt chuẩn (Tốt - Loại 1) |
| Banana Spoiled | Chuối · Bị hỏng | `chuoi-hong` | Không đạt chuẩn (Kém - Loại bỏ) |
| Mango Unripe | Xoài · Chưa chín | `xoai-chua-chin` | Đạt chuẩn tiêu thụ (Tốt - Loại 2) |
| Mango Ripe | Xoài · Chín | `xoai-chin` | Đạt chuẩn (Tốt - Loại 1) |
| Mango Spoiled | Xoài · Bị hỏng | `xoai-hong` | Không đạt chuẩn (Kém - Loại bỏ) |

Dataset là dữ liệu ảnh thực tế cục bộ, bị Git bỏ qua (`.gitignore`) và không commit vào repository.

---

## 2. Dataset YOLO cũ (Dữ liệu lịch sử tham khảo)

Thư mục `data/raw/` trước đây lưu dataset YOLO phát hiện đối tượng với nhãn Apple/Banana/Mango và các trạng thái `Unripe`, `Ripe`, `Overripe`. Đây là dữ liệu gốc từng được dùng để tạo dataset ảnh nguyên cảnh `data/processed/whole-scene/` trong các thử nghiệm ban đầu.

### Cấu trúc và số lượng dữ liệu YOLO cũ (Tham khảo)

Pipeline trích xuất ảnh toàn cảnh trước đây dùng lệnh:

```powershell
python -m src.dataset --source-dir data/raw --output-dir data/processed/whole-scene
```

`data.yaml` và `class_mapping.json` chỉ mô tả dataset YOLO cũ.

### Phân hoạch và số lượng YOLO cũ

| Tập | Ảnh | Tệp nhãn |
|---|---:|---:|
| Train | 1.559 | 1.559 |
| Validation | 445 | 445 |
| Test | 223 | 223 |
| **Tổng** | **2.227** | **2.227** |

Mỗi tệp nhãn có cùng tên gốc với ảnh tương ứng, nằm trong thư mục `labels/`
cùng tập. Mỗi dòng nhãn theo định dạng YOLO:

```text
class_id x_center y_center width height
```

Các tọa độ được chuẩn hóa theo kích thước ảnh. Thống kê dưới đây đếm **đối
tượng được gán nhãn**, không phải số ảnh của mỗi lớp:

| ID | Lớp | Đối tượng |
|---:|---|---:|
| 0 | Apple Unripe | 256 |
| 1 | Apple Ripe | 279 |
| 2 | Apple Overripe | 251 |
| 3 | Banana Unripe | 312 |
| 4 | Banana Ripe | 372 |
| 5 | Banana Overripe | 255 |
| 6 | Mango Unripe | 261 |
| 7 | Mango Ripe | 269 |
| 8 | Mango Overripe | 252 |

### Ánh xạ và giới hạn nhãn YOLO cũ

Các ID đích trong bảng trên khớp với thứ tự `names` trong `data.yaml`.
`class_mapping.json` cho biết lớp nguồn nào được giữ lại; các nhãn `Rotten` và
các loại trái cây ngoài táo, chuối, xoài không thuộc 9 lớp đích. Không suy diễn
rằng dataset hỗ trợ các lớp đã bị loại khỏi ánh xạ.

Đây là dữ liệu **phát hiện đối tượng**, không phải cấu trúc phân loại ảnh theo
thư mục. Khi dùng, giữ nguyên các phân hoạch đã có và không dùng tập test để
huấn luyện, chọn mô hình hoặc điều chỉnh tham số. Nếu tạo thêm tập dữ liệu hay
thay đổi ánh xạ nhãn, cập nhật đồng bộ `data.yaml`, `class_mapping.json` và tài
liệu này.

### Nguồn và giấy phép YOLO cũ

Dataset **Fruit Ripeness** được thu thập từ [Roboflow Universe](https://universe.roboflow.com/huzaifa-qgpco/fruit-ripeness-unjex-xsudt),
do tác giả Huzaifa công bố. Dữ liệu được sử dụng theo giấy phép
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Giấy phép cho phép sao chép, chia sẻ, chỉnh sửa và sử dụng cho mục đích thương
mại hoặc phi thương mại, với điều kiện ghi nhận tác giả, dẫn nguồn dataset,
cung cấp liên kết đến giấy phép và nêu rõ các thay đổi đã thực hiện (nếu có).

Khi phân phối lại dữ liệu hoặc bản phái sinh, hãy giữ thông tin ghi công và
thông báo các thay đổi phù hợp với điều khoản CC BY 4.0.

## Dataset YOLO ảnh toàn cảnh

Tạo dữ liệu bằng lệnh:

```powershell
python -m src.dataset --source-dir data/raw --output-dir data/processed/whole-scene
```

Pipeline sao chép ảnh gốc, không crop bounding box và giữ nguyên phân hoạch
train/val/test YOLO để tránh trộn các split. Một ảnh chỉ có thể có một nhãn
classification nếu tất cả đối tượng trong ảnh cùng thuộc một class. Ảnh không
có nhãn, nhãn lỗi hoặc có nhiều class bị loại; số lượng theo split/lớp và lý do
bị loại được ghi trong `data/processed/whole-scene/manifest.json`.

Với dataset YOLO cũ, export giữ 1.547 ảnh train, 443 validation và 220 test;
17 ảnh bị loại do có nhiều class trong cùng ảnh. Kết quả train hiện tại được
lưu riêng trong `models-whole-scene/` và `reports-whole-scene/`; đây là kết quả
của taxonomy Apple/Banana/Mango cũ.

Train dùng ảnh nguyên cảnh với `python -m src.train`.
Dataset đã xuất và manifest là dữ liệu tạo sinh cục bộ, không commit vào
repository.
