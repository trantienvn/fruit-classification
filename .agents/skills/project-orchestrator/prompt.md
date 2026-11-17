Sử dụng skill `project-orchestrator`.

Trước khi thực hiện:
1. Đọc các tài liệu liên quan trong thư mục `context/` để nắm lịch sử, kiến trúc, quyết định và tiến độ dự án.
2. Kiểm tra hoặc tạo `context/task-board.md` nếu chưa tồn tại.
3. Xác định Agent ID của bạn.
4. Kiểm tra task đang được giao, trạng thái và các dependency.
5. Đăng ký task phù hợp với yêu cầu hiện tại trước khi sửa code.
6. Không làm trùng hoặc ghi đè phần việc của agent khác.

Sau khi hoàn thành:
- Chạy kiểm tra phù hợp.
- Cập nhật trạng thái task.
- Đồng bộ các quyết định, thay đổi và tiến độ quan trọng vào `context/`.
