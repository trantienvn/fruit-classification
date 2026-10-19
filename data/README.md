# Dữ liệu

Hiện chưa có dataset trong workspace. Đặt dữ liệu cục bộ dưới `raw/`; nội dung ảnh và dữ liệu sinh ra không được commit (xem `.gitignore`).

## Định dạng đề xuất

```text
data/raw/
  train/
    apple/ripe/good/*.jpg
    apple/unripe/damaged/*.jpg
    banana/ripe/good/*.png
  validation/
    <fruit>/<ripeness>/<quality>/<image>.<jpg|png>
  test/
    <fruit>/<ripeness>/<quality>/<image>.<jpg|png>
```

Tên thư mục là nhãn cho ba tác vụ: loại trái cây, độ chín, chất lượng. Dùng cùng một danh mục nhãn ở cả ba tập; chọn tên nhãn rõ ràng, nhất quán (ví dụ `unripe`, `ripe`, `overripe`; `good`, `damaged`). Ảnh phải đọc được bởi Pillow. Mỗi mẫu chỉ thuộc một tổ hợp nhãn. Nếu có nhãn thiếu, nhiều nhãn hoặc nhãn dạng mức độ, hãy mô tả điều đó trước khi chọn cấu hình mô hình.

Giữ các ảnh của cùng một trái/quả hoặc phiên chụp trong cùng một phân hoạch. Ghi nguồn dữ liệu, giấy phép, quy trình gán nhãn, số lượng mẫu từng lớp và các hạn chế. Tránh rò rỉ giữa train/validation/test.

