# Dữ liệu

`data/raw/` hiện chứa dataset ảnh trái cây theo định dạng YOLO, với nhãn phát
hiện đối tượng theo loại trái cây và độ chín. Dataset có ba tập `train`, `val`,
`test` và 9 lớp: táo, chuối, xoài; mỗi loại gồm `Unripe`, `Ripe`, `Overripe`.

## Cấu trúc hiện tại

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
```

`data.yaml` khai báo đường dẫn tương đối đến các tập ảnh, số lớp (`nc: 9`) và
tên lớp. `class_mapping.json` ghi lại ánh xạ từ lớp nguồn sang 9 lớp đang dùng;
`balanced_distribution.png` là biểu đồ phân phối lớp.

## Phân hoạch và số lượng

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

## Ánh xạ và giới hạn nhãn

Các ID đích trong bảng trên khớp với thứ tự `names` trong `data.yaml`.
`class_mapping.json` cho biết lớp nguồn nào được giữ lại; các nhãn `Rotten` và
các loại trái cây ngoài táo, chuối, xoài không thuộc 9 lớp đích. Không suy diễn
rằng dataset hỗ trợ các lớp đã bị loại khỏi ánh xạ.

Đây là dữ liệu **phát hiện đối tượng**, không phải cấu trúc phân loại ảnh theo
thư mục. Khi dùng, giữ nguyên các phân hoạch đã có và không dùng tập test để
huấn luyện, chọn mô hình hoặc điều chỉnh tham số. Nếu tạo thêm tập dữ liệu hay
thay đổi ánh xạ nhãn, cập nhật đồng bộ `data.yaml`, `class_mapping.json` và tài
liệu này.

## Nguồn và giấy phép

Dataset **Fruit Ripeness** được thu thập từ [Roboflow Universe](https://universe.roboflow.com/huzaifa-qgpco/fruit-ripeness-unjex-xsudt),
do tác giả Huzaifa công bố. Dữ liệu được sử dụng theo giấy phép
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Giấy phép cho phép sao chép, chia sẻ, chỉnh sửa và sử dụng cho mục đích thương
mại hoặc phi thương mại, với điều kiện ghi nhận tác giả, dẫn nguồn dataset,
cung cấp liên kết đến giấy phép và nêu rõ các thay đổi đã thực hiện (nếu có).

Khi phân phối lại dữ liệu hoặc bản phái sinh, hãy giữ thông tin ghi công và
thông báo các thay đổi phù hợp với điều khoản CC BY 4.0. Hiện
`data/processed/` chưa có dữ liệu đã xử lý.
