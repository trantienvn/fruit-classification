# Project Context — Phân loại chất lượng và độ chín trái cây

Thư mục này lưu trữ toàn bộ ngữ cảnh, quyết định kiến trúc, tiến độ và lịch sử giải quyết vấn đề của dự án theo quy chuẩn **Context Manager Skill** (`.agents/skills/context-manager/SKILL.MD`).

## Mục đích
- Duy trì kiến thức dự án giữa các phiên làm việc (work sessions).
- Tránh trùng lặp công việc và không phá vỡ các tính năng đã hoạt động ổn định.
- Lưu lại các quyết định kỹ thuật đã được người dùng xác nhận.

## Cấu trúc tài liệu

| Tệp tin | Nội dung chính |
|---|---|
| [`project-overview.md`](project-overview.md) | Mục tiêu, công nghệ, phạm vi 9 lớp, thành phần cốt lõi của đề tài |
| [`architecture.md`](architecture.md) | Cấu trúc thư mục, luồng dữ liệu (Data Pipeline, Feature Extraction, Inference), UI |
| [`decisions.md`](decisions.md) | Các quyết định kỹ thuật đã chốt (Python 3.14, 9 lớp kết hợp, crop bbox YOLO, bỏ DL) |
| [`progress.md`](progress.md) | Trạng thái từng giai đoạn (Đã hoàn thành, đang làm dở, việc tiếp theo) |
| [`issues.md`](issues.md) | Lỗi gặp phải, nguyên nhân, cách xử lý và bài học kinh nghiệm |
| [`changelog.md`](changelog.md) | Nhật ký thay đổi mã nguồn chi tiết theo thời gian |
| [`sessions/`](sessions/) | Nhật ký chi tiết của từng phiên làm việc (định dạng `YYYY-MM-DD.md`) |

## Quy tắc cập nhật
- **Đọc trước khi làm**: Mọi phiên làm việc mới đều phải đọc tài liệu trong thư mục này trước khi sửa code.
- **Bảo toàn dữ liệu**: Không xóa, ghi đè thông tin lịch sử khi chưa kiểm chứng.
- **Cập nhật sau phiên làm việc**: Ghi nhận tiến độ mới, lỗi phát sinh và cập nhật `sessions/YYYY-MM-DD.md`.

