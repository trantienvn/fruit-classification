# API Contract — Web Application

## Trạng thái và nguồn chuẩn

- Trạng thái: **CONFIRMED — TASK-H01, 2026-10-31**.
- Runtime: FastAPI cùng origin với UI, mặc định `http://127.0.0.1:8000`.
- Nguồn chuẩn: route và Pydantic schema trong `app/web/`; OpenAPI runtime tại `/openapi.json`.
- Snapshot có thể review/đưa vào tooling: `app/web/openapi.json`, sinh bằng `python -m app.web.export_openapi`.
- `tests/test_web_api.py` xác nhận snapshot khớp schema runtime.

Không yêu cầu xác thực cho ứng dụng demo cục bộ. UI gọi API cùng origin; không bật CORS.

## Endpoints

| Method và path | Input | Kết quả |
|---|---|---|
| `GET /api/health` | Không có | `status`, `model_ready`, `history_ready`, `model_name` khi có model. HTTP 200 phản ánh tiến trình còn hoạt động; `status` có thể là `degraded`. |
| `POST /api/predict` | Multipart field `file` | Dự đoán bằng `FruitClassifier.predict_from_bytes()` và ghi metadata vào lịch sử. |
| `GET /api/model-info` | Không có | Metadata từ classifier/config: model, nhãn, số feature, PCA và metrics đã lưu. HTTP 503 nếu model chưa sẵn sàng. |
| `GET /api/history` | `q`, `class_name`, `limit` 1–100, `offset` ≥ 0 | Danh sách mới nhất trước, kèm `total`, `limit`, `offset`. |
| `DELETE /api/history` | Không có | Xóa các record SQLite hiện có, trả số record đã xóa. |
| `GET /api/statistics` | Không có | Đếm lượt phân loại đã lưu theo nhãn, loại quả, trạng thái và ngày trong 7 ngày gần nhất. |

## Prediction contract

Upload chấp nhận JPG/JPEG, PNG, BMP và WEBP, tối đa 10 MiB. Server kiểm tra phần mở rộng và giải mã ảnh thực tế. Lỗi trả JSON `{success: false, error: {code, message}}` với HTTP 413 (quá lớn), 415 (định dạng không hỗ trợ), 422 (ảnh/request không hợp lệ) hoặc 503 (model chưa sẵn sàng).

Response thành công có `success: true` và `data` gồm:

- `id`, `filename`, `predicted_class`, `class_name`, `class_name_vi`, `fruit_type`, `ripeness`.
- `class_name` giữ tên lớp canonical từ artifact model; model hiện hành dùng Cam/Chuối/Xoài × Unripe/Ripe/Spoiled. `class_name_vi` trả nhãn đầy đủ bằng tiếng Việt (ví dụ `Cam · Hỏng`). Lịch sử cũ cũng được giữ và serialize, không cần đổi schema SQLite.
- `processing_time_ms` và `created_at` UTC.
- `confidence` và `probabilities` chỉ có mặt nếu classifier có `predict_proba`. Nếu không, API bỏ hai trường đó; không chuyển decision score thành xác suất.

Không có trường “quality grade”: model được huấn luyện cho 9 lớp loại quả × trạng thái, không có nhãn chất lượng thương mại.

## Lưu trữ và dữ liệu thống kê

SQLite ở `app/web/data/history.sqlite3` lưu tên file và metadata dự đoán. Ảnh, nội dung bytes và vector xác suất không được lưu. `GET /api/statistics` tính từ các lượt đã ghi; khi chưa có lượt nào, tổng bằng 0 và các nhãn trả count 0. Nhãn legacy trong lịch sử (ví dụ Apple từ model cũ) vẫn được tính riêng thay vì làm hỏng thống kê. Metrics kiểm thử của model được đọc riêng từ `models/config.json`.

## Giới hạn và dependency

- API và UI phải được phục vụ cùng origin; triển khai nhiều origin cần thiết kế CORS riêng.
- Upload file lớn bị từ chối trước inference; không có xử lý batch.
- Dùng FastAPI, Uvicorn và `python-multipart`; history dùng SQLite trong Python standard library.
- Các endpoint và schemas phải được cập nhật cùng `app/web/openapi.json`, docs và tests khi contract đổi.
