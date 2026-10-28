# Dữ liệu

Thư mục `data/raw/` hiện có các bộ dữ liệu ảnh về táo và chuối. Đây là các nguồn
dữ liệu riêng, phục vụ những tác vụ và cách gán nhãn khác nhau; không gộp chúng
thành một dataset đa nhãn nếu chưa chuẩn hóa nhãn và xác nhận mục tiêu huấn luyện.
`data/processed/` dành cho dữ liệu đã xử lý; hiện thư mục này chưa có dữ liệu.

## Cấu trúc hiện tại

```text
data/
  raw/
    Apple/
      Training/Apple/<variety>/
      Test/Apple/<variety>/
      README.md
      LICENSE
    mango-banana/
      Banana Classification Dataset/<variety>/
      Augmented Banana Classification Dataset/<variety>/
      Banana Ripeness Detection Dataset/<ripeness>/
      Augmented Banana Ripeness Detection Dataset/<ripeness>/
      train/
        images/
        labels/
      test/
        images/
        labels/
  processed/
```

### Apple

Ảnh được chia sẵn thành `Training` và `Test`, bên dưới là các thư mục theo
giống táo. Bộ dữ liệu tại đây là phần dữ liệu Apple; không suy ra rằng các lớp
cherries và tomatoes được mô tả trong tài liệu nguồn cũng có trong bản sao này.
Thông tin nguồn và giấy phép được lưu tại `data/raw/Apple/README.md` và
`data/raw/Apple/LICENSE`. Tài liệu nguồn ghi giấy phép CC BY-SA 4.0; cần giữ
ghi công và tuân thủ điều khoản ShareAlike khi chia sẻ bản phái sinh.

### Banana classification và ripeness

Hai thư mục `Banana Classification Dataset` và `Augmented Banana Classification
Dataset` tổ chức ảnh theo giống chuối. Hai thư mục `Banana Ripeness Detection
Dataset` và `Augmented Banana Ripeness Detection Dataset` tổ chức ảnh theo mức
độ chín (`Green`, `Semi-ripe`, `Ripe`, `Overripe`). Các phiên bản augmented là
các bộ riêng; không gộp chúng với bản gốc trước khi chia tập nếu chưa kiểm tra
trùng lặp hoặc nguồn gốc của ảnh.

### Bộ `train`/`test` có nhãn detection

`mango-banana/train/` và `mango-banana/test/` chứa `images/` cùng `labels/`;
mỗi thư mục nhãn có `classes.txt` và các tệp nhãn tương ứng với ảnh. Đây là cấu
trúc khác với phân loại theo thư mục. Trước khi dùng, đối chiếu `classes.txt`,
đảm bảo ánh xạ nhãn khớp giữa train và test, và kiểm tra định dạng nội dung tệp
nhãn với công cụ huấn luyện dự định sử dụng.

## Lưu ý khi sử dụng

- Chọn một nguồn và một mục tiêu cụ thể: phân loại giống, phân loại độ chín
  hoặc phát hiện đối tượng. Các nhãn trong những bộ trên không tương đương nhau.
- Giữ nguyên các phân hoạch `Training`/`Test` hoặc `train`/`test` do nguồn cung
  cấp. Không dùng tập test để huấn luyện, chọn mô hình hay điều chỉnh tham số.
- Với ảnh tăng cường, kiểm tra ảnh gốc và ảnh biến đổi không bị chia sang các
  tập khác nhau; việc tăng cường chỉ nên thực hiện trên tập huấn luyện.
- Kiểm tra ảnh đọc được, nhãn hợp lệ, số lượng mẫu mỗi lớp và các tệp bị thiếu
  trước khi chạy pipeline. Ghi rõ nguồn, giấy phép và mọi thay đổi dữ liệu.
- Thông tin nguồn/giấy phép của các bộ trong `mango-banana/` chưa được ghi lại
  tại đây. Hãy xác minh điều khoản từ nguồn tải trước khi phân phối hoặc sử
  dụng lại chúng.
- Không thêm dữ liệu, trọng số mô hình hoặc kết quả huấn luyện giả vào
  repository. Dữ liệu đã xử lý đặt trong `data/processed/`.
