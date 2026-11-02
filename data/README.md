# Dữ liệu

## Dataset classification hiện hành

Pipeline mặc định dùng dataset ảnh đã chia lớp tại
`data/processed/dataset/{train,val,test}/`. Dataset có 9 lớp: cam, chuối, xoài;
mỗi loại gồm `Unripe`, `Ripe`, `Spoiled`. Các thư mục `*-hong` được xác nhận là
trái cây hỏng, không gộp vào nhãn quá chín.

| Tập | Ảnh |
|---|---:|
| Train | 1.270 |
| Validation | 361 |
| Test | 184 |
| **Tổng** | **1.815** |

Ánh xạ tên thư mục sang nhãn model:

| Nhãn model | Thư mục |
|---|---|
| Orange Unripe | `cam-chua-chin` |
| Orange Ripe | `cam-chin` |
| Orange Spoiled | `cam-hong` |
| Banana Unripe | `chuoi-chua-chin` |
| Banana Ripe | `chuoi-chin` |
| Banana Spoiled | `chuoi-hong` |
| Mango Unripe | `xoai-chua-chin` |
| Mango Ripe | `xoai-chin` |
| Mango Spoiled | `xoai-hong` |

Dataset là dữ liệu cục bộ, bị Git bỏ qua và không được thêm vào repository.
Nguồn/giấy phép của bộ thư mục này chưa được ghi trong repository; cần giữ
thông tin nguồn và giấy phép gốc khi chia sẻ dữ liệu.

## Dataset YOLO cũ

`data/raw/` vẫn giữ dataset YOLO cũ với nhãn Apple/Banana/Mango và các trạng
thái `Unripe`, `Ripe`, `Overripe`. Dataset này chỉ dùng khi chạy rõ
`--dataset-mode yolo` hoặc tạo ảnh toàn cảnh theo chế độ tương thích; không
được dùng làm dataset mặc định Cam/Chuối/Xoài.

### Cấu trúc YOLO cũ

```text
data/
  raw/
    data.yaml
    class_mapping.json
    balanced_distribution.png
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
  processed/
    dataset/
      train/{cam-chin,...}/
      val/{cam-chin,...}/
      test/{cam-chin,...}/
    whole-scene/
      manifest.json
      train/{Apple_Unripe,...}/
      val/{Apple_Unripe,...}/
      test/{Apple_Unripe,...}/
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

Để train trên ảnh nguyên cảnh, dùng
`python -m src.train --dataset-mode whole-scene --data-dir data/processed/whole-scene`.
Dataset đã xuất và manifest là dữ liệu tạo sinh cục bộ, không commit vào
repository.
