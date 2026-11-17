# Hướng dẫn làm việc trong dự án

- Dùng Python 3.14 và giữ mã nguồn, tài liệu bằng tiếng Việt khi phù hợp.
- Không thêm dataset, trọng số mô hình, hoặc kết quả huấn luyện giả vào repository.
- Dữ liệu thật đặt dưới `data/raw/`; cấu trúc nhãn được mô tả trong `data/README.md`.
- Tách tiền xử lý, huấn luyện, đánh giá và dự đoán thành các mô-đun trong `src/`.
- Không tự tải dữ liệu lớn hay cài thư viện nặng. Chỉ tải trọng số pretrained khi người dùng chủ động chạy quy trình huấn luyện.
- Giữ các bài kiểm thử độc lập với dataset và không yêu cầu tải mô hình.

