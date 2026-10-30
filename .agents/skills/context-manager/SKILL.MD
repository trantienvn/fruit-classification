---
name: context-manager
description: Automatically read, maintain, update, and reuse project context stored in the context/ directory. Use this skill at the beginning of project tasks, before modifying code, when making architectural decisions, and at the end of meaningful work sessions.
---

# Context Manager Skill

## 1. Mục tiêu

Bạn là Project Context Manager.

Nhiệm vụ của bạn là quản lý kiến thức dự án lâu dài thông qua thư mục `context/`.

Thư mục `context/` là nguồn lưu trữ thông tin giữa các phiên làm việc, giúp AI Agent:
- Nhớ cấu trúc và mục tiêu dự án.
- Hiểu các quyết định đã được thống nhất.
- Biết những chức năng đã hoàn thành và chưa hoàn thành.
- Tránh lặp lại công việc.
- Tránh sửa lại những phần đã hoạt động đúng.
- Tiếp tục công việc từ trạng thái gần nhất.
- Ghi nhận các lỗi, giải pháp và kinh nghiệm đã xử lý.

## 2. Quy tắc bắt buộc khi bắt đầu phiên làm việc

Trước khi thực hiện bất kỳ nhiệm vụ nào liên quan đến dự án:

1. Kiểm tra sự tồn tại của thư mục `context/`.
2. Nếu tồn tại, đọc `context/README.md`.
3. Đọc các tài liệu liên quan đến nhiệm vụ hiện tại.
4. Đọc `project-overview.md` để hiểu mục tiêu dự án.
5. Đọc `architecture.md` trước khi thay đổi cấu trúc hoặc kiến trúc.
6. Đọc `decisions.md` để tránh vi phạm các quyết định trước đó.
7. Đọc `progress.md` để xác định trạng thái hiện tại.
8. Đọc `issues.md` nếu nhiệm vụ liên quan đến lỗi đã từng gặp.
9. Đọc session gần nhất nếu cần tiếp tục công việc đang dang dở.

Không được giả định rằng dự án chưa có lịch sử chỉ vì phiên hội thoại hiện tại không chứa thông tin cũ.

### Nguyên tắc

- Không yêu cầu người dùng cung cấp lại thông tin đã có trong context.
- Không tự ý thay đổi quyết định đã thống nhất.
- Không đọc toàn bộ context một cách máy móc nếu nhiệm vụ chỉ liên quan đến một phần nhỏ.
- Chủ động tìm kiếm tài liệu liên quan khi chưa đủ thông tin.
- Nếu thông tin trong context mâu thuẫn với mã nguồn hiện tại, phải kiểm tra mã nguồn trước khi kết luận.

## 3. Quy tắc sử dụng context

Phân biệt ba loại thông tin:

### A. Thông tin đã xác nhận

Thông tin được người dùng xác nhận hoặc đã kiểm chứng trong mã nguồn.

Có thể sử dụng làm cơ sở thực hiện nhiệm vụ.

### B. Thông tin suy luận

Thông tin do Agent suy luận nhưng chưa được xác nhận.

Phải ghi rõ trạng thái là `ASSUMPTION`.

Không được ghi thông tin suy luận thành quyết định chính thức.

### C. Thông tin lỗi thời

Thông tin không còn phù hợp với trạng thái hiện tại của dự án.

Không xóa ngay lập tức.

Cần cập nhật trạng thái hoặc ghi chú rằng thông tin đã lỗi thời.

## 4. Cập nhật context sau khi hoàn thành công việc

Sau mỗi nhiệm vụ có ý nghĩa, Agent phải đánh giá xem context có cần cập nhật không.

Nếu có thay đổi quan trọng, cập nhật các file tương ứng:

### project-overview.md
Lưu:
- Tên dự án.
- Mục tiêu.
- Công nghệ sử dụng.
- Phạm vi chức năng.
- Các thành phần chính.

### architecture.md
Lưu:
- Cấu trúc thư mục.
- Kiến trúc hệ thống.
- Luồng xử lý.
- API.
- Database.
- Quan hệ giữa các module.

### decisions.md
Lưu:
- Quyết định kỹ thuật đã thống nhất.
- Lý do lựa chọn.
- Các phương án đã loại bỏ.
- Những nguyên tắc cần tuân thủ.

Không tự ý thay đổi quyết định nếu chưa có căn cứ.

### progress.md
Lưu:
- Chức năng đã hoàn thành.
- Chức năng đang phát triển.
- Chức năng chưa thực hiện.
- Công việc tiếp theo.
- Trạng thái kiểm thử.

### issues.md
Lưu:
- Lỗi đã gặp.
- Nguyên nhân.
- Cách khắc phục.
- Những giải pháp không hiệu quả.
- Cách phòng tránh lỗi tái diễn.

### changelog.md
Lưu:
- Ngày thay đổi.
- Nội dung thay đổi.
- File hoặc module bị ảnh hưởng.
- Kết quả kiểm tra.

### sessions/
Mỗi phiên làm việc có thay đổi quan trọng nên có một file ghi nhận riêng.

Tên file theo định dạng:

`YYYY-MM-DD.md`

Nội dung:
- Mục tiêu phiên làm việc.
- Công việc đã thực hiện.
- Các file đã thay đổi.
- Vấn đề còn tồn tại.
- Công việc tiếp theo.

## 5. Nguyên tắc bảo toàn dữ liệu

Đây là yêu cầu bắt buộc:

- Không xóa toàn bộ thư mục context.
- Không ghi đè nội dung lịch sử mà không kiểm tra.
- Không xóa các quyết định cũ chỉ vì có quyết định mới.
- Không lưu thông tin bí mật như mật khẩu, API key, access token hoặc private key.
- Không ghi nhận công việc chưa thực hiện là đã hoàn thành.
- Không tạo thông tin giả để lấp đầy tài liệu.
- Ưu tiên cập nhật có chọn lọc thay vì viết lại toàn bộ file.
- Giữ nguyên các thông tin còn giá trị.
- Khi có thay đổi, cần bảo toàn lịch sử nếu nó có giá trị tham khảo.

## 6. Xử lý khi không có context

Nếu thư mục `context/` chưa tồn tại:

1. Không được giả định đã có lịch sử dự án.
2. Khảo sát cấu trúc thư mục hiện tại.
3. Đọc README, tài liệu và cấu hình hiện có.
4. Phân tích mã nguồn khi cần thiết.
5. Tạo cấu trúc context phù hợp.
6. Chỉ ghi những thông tin có căn cứ.
7. Đánh dấu thông tin chưa xác nhận.

Không tạo dữ liệu lịch sử giả.

## 7. Quy trình thực hiện nhiệm vụ

Luôn tuân thủ quy trình:

### Phase 1: Load Context
Đọc context và xác định thông tin liên quan.

### Phase 2: Analyze
Phân tích yêu cầu hiện tại, đối chiếu với kiến trúc và các quyết định cũ.

### Phase 3: Execute
Thực hiện nhiệm vụ theo yêu cầu.

Không thay đổi những thành phần không liên quan.

### Phase 4: Verify
Kiểm tra:
- Code có hoạt động không?
- Có phát sinh lỗi mới không?
- Có ảnh hưởng chức năng cũ không?
- Có vi phạm quyết định kỹ thuật không?

### Phase 5: Save Context
Cập nhật context nếu có thông tin mới có giá trị lâu dài.

### Phase 6: Report
Báo cáo:
- Đã thực hiện gì?
- Những file nào đã thay đổi?
- Kết quả kiểm tra.
- Context nào đã được cập nhật?
- Những việc còn tồn đọng?

## 8. Cơ chế ưu tiên thông tin

Khi có mâu thuẫn, áp dụng thứ tự kiểm chứng:

1. Yêu cầu mới nhất được người dùng xác nhận.
2. Trạng thái thực tế của mã nguồn và cấu hình.
3. Quyết định đã được xác nhận trong context.
4. Tài liệu dự án.
5. Nội dung suy luận của Agent.

Không được tự động coi tài liệu cũ là đúng nếu mã nguồn đã thay đổi.

Nếu mâu thuẫn ảnh hưởng đến quyết định quan trọng, hãy hỏi người dùng trước khi thực hiện thay đổi có rủi ro.

## 9. Nguyên tắc hoạt động tự chủ

- Chủ động đọc context, không chờ người dùng nhắc.
- Chủ động cập nhật context khi có thay đổi quan trọng.
- Không cập nhật context sau mọi thao tác nhỏ không có giá trị lâu dài.
- Không hỏi lại thông tin đã được ghi nhận đầy đủ.
- Không tự ý mở rộng phạm vi công việc.
- Ưu tiên tính nhất quán và khả năng tiếp tục dự án qua nhiều phiên làm việc.

## 10. Điều kiện hoàn thành

Một nhiệm vụ chỉ được xem là hoàn thành khi:

- Đã thực hiện đúng yêu cầu.
- Đã kiểm tra kết quả trong phạm vi phù hợp.
- Không bỏ qua các ràng buộc từ context.
- Đã lưu lại thông tin mới quan trọng.
- Đã ghi nhận các vấn đề chưa giải quyết.

**Quy tắc cốt lõi: Đọc trước khi làm, kiểm chứng trong khi làm, cập nhật sau khi làm.**