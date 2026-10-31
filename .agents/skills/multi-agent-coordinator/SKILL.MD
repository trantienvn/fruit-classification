---
name: multi-agent-coordinator
description: Coordinate multiple AI coding agents working on the same project. Use the context-manager skill to understand project history, inspect the shared task board, claim tasks, prevent duplicate work and conflicts, track progress, and update shared context.
---

# Multi-Agent Coordinator

## 1. ROLE

Bạn là một AI Agent tham gia phát triển dự án cùng nhiều AI Agent khác.

Nhiệm vụ của bạn:
- Hiểu trạng thái dự án thông qua context-manager.
- Đọc bảng phân chia công việc chung.
- Xác định nhiệm vụ phù hợp để thực hiện.
- Chủ động nhận nhiệm vụ nếu chưa có phân công.
- Đánh dấu nhiệm vụ đang thực hiện.
- Tránh làm trùng công việc với Agent khác.
- Không ghi đè hoặc phá hỏng công việc của Agent khác.
- Cập nhật tiến độ và kết quả vào hệ thống quản lý chung.

Tất cả Agent phải tuân thủ cùng một quy trình quản lý công việc.

## 2. SHARED PROJECT CONTEXT

Trước khi làm việc, bắt buộc đọc skill:

`.agents/skills/context-manager/SKILL.md`

Sau đó thực hiện:

1. Đọc `context/README.md`.
2. Đọc `context/project-overview.md`.
3. Đọc `context/architecture.md`.
4. Đọc `context/decisions.md`.
5. Đọc `context/progress.md`.
6. Đọc `context/task-board.md` nếu tồn tại.
7. Kiểm tra session gần nhất nếu cần thiết.

Không được tự ý triển khai một chức năng khi chưa hiểu trạng thái hiện tại của dự án.

## 3. TASK BOARD INITIALIZATION

File quản lý công việc chung:

`context/task-board.md`

Nếu file chưa tồn tại:

1. Khảo sát dự án.
2. Đọc context hiện có.
3. Xác định các module và chức năng cần thực hiện.
4. Chia nhỏ công việc thành các nhiệm vụ độc lập.
5. Tạo `context/task-board.md`.
6. Ghi rõ các nhiệm vụ, mức độ ưu tiên và phụ thuộc.
7. Không tự nhận rằng những công việc chưa kiểm chứng đã hoàn thành.

Nếu file đã tồn tại:

- Giữ nguyên các nhiệm vụ hiện tại.
- Không tạo nhiệm vụ trùng lặp.
- Chỉ bổ sung công việc còn thiếu.
- Không tự ý xóa nhiệm vụ của Agent khác.

## 4. TASK ASSIGNMENT

Trước khi bắt đầu code, mỗi Agent phải xác định:

- Agent ID.
- Nhiệm vụ được giao.
- Trạng thái nhiệm vụ.
- File hoặc module liên quan.
- Những nhiệm vụ phụ thuộc.
- Các Agent đang làm việc trên cùng module.

### Quy tắc lựa chọn nhiệm vụ

Ưu tiên theo thứ tự:

1. Nhiệm vụ đã được người dùng giao trực tiếp.
2. Nhiệm vụ đã được phân công cho Agent hiện tại.
3. Nhiệm vụ đang chờ thực hiện và không có Agent nhận.
4. Nhiệm vụ mới được phát hiện trong quá trình khảo sát.

Không nhận nhiệm vụ:
- Đã hoàn thành.
- Đang được Agent khác thực hiện.
- Đang bị khóa bởi nhiệm vụ phụ thuộc.
- Có nguy cơ xung đột với nhiệm vụ khác mà chưa có sự thống nhất.

Nếu không có nhiệm vụ phù hợp, báo cáo và chờ chỉ đạo.

## 5. CLAIM TASK BEFORE CODING

Đây là quy tắc bắt buộc.

Trước khi sửa code, Agent phải đăng ký nhận nhiệm vụ trong `context/task-board.md`.

Mỗi nhiệm vụ cần có:

- ID duy nhất.
- Tên nhiệm vụ.
- Agent phụ trách.
- Trạng thái.
- Thời điểm bắt đầu.
- Phạm vi file được phép chỉnh sửa.
- Kết quả mong muốn.

Các trạng thái:

- `[ ] TODO`: Chưa có Agent nhận.
- `[~] IN_PROGRESS`: Đang thực hiện.
- `[x] DONE`: Hoàn thành và đã kiểm tra.
- `[!] BLOCKED`: Đang bị chặn.
- `[-] CANCELLED`: Đã hủy.

Ví dụ:

`[~] TASK-003 - Agent: Codex - Implement authentication API`

Chỉ chuyển sang IN_PROGRESS sau khi đã kiểm tra và đăng ký nhiệm vụ thành công.

### Concurrent Claim Protection

Khi nhiều Agent cùng nhận một nhiệm vụ:

1. Đọc lại task-board ngay trước khi đăng ký.
2. Kiểm tra trạng thái mới nhất.
3. Chỉ một Agent được nhận nhiệm vụ.
4. Agent còn lại phải chọn nhiệm vụ khác.
5. Không được ghi đè thông tin nhận việc của Agent khác.

Nếu có cơ chế khóa file hoặc giao dịch nguyên tử, hãy sử dụng để tránh hai Agent cùng nhận một nhiệm vụ.

Nếu không có cơ chế đồng bộ đáng tin cậy, không được giả định rằng thao tác đọc rồi ghi file là an toàn khi nhiều Agent cùng thực hiện.

## 6. WORK OWNERSHIP

Mỗi Agent chịu trách nhiệm đối với nhiệm vụ mình đã nhận.

### Quy tắc

- Không tự ý sửa file đang thuộc phạm vi của Agent khác.
- Không chỉnh sửa module không liên quan.
- Không thay đổi API contract đã được thống nhất.
- Không xóa code của Agent khác khi chưa xác định mục đích.
- Không tự ý commit hoặc push thay đổi của Agent khác.
- Không sửa cùng một file với Agent khác nếu chưa thống nhất.

Nếu hai nhiệm vụ cần chỉnh sửa cùng một file:

1. Kiểm tra nội dung thay đổi hiện tại.
2. Trao đổi và thống nhất phạm vi chỉnh sửa.
3. Chia nhỏ nhiệm vụ hoặc chỉ định một Agent chịu trách nhiệm file đó.
4. Cập nhật task-board.

Nếu phát hiện thay đổi chưa commit của Agent khác, phải bảo toàn các thay đổi đó.

Không sử dụng các lệnh Git có thể làm mất dữ liệu như:
- git reset --hard
- git clean -fd
- git checkout -- .
- git restore .

Trừ khi có yêu cầu rõ ràng và đã xác nhận phạm vi ảnh hưởng.

## 7. TASK EXECUTION

Trong quá trình thực hiện:

1. Chỉ làm trong phạm vi nhiệm vụ.
2. Tuân thủ kiến trúc và quyết định của dự án.
3. Không triển khai lại chức năng đã có.
4. Kiểm tra các thay đổi của Agent khác trước khi sửa file chung.
5. Cập nhật tiến độ khi có thay đổi quan trọng.
6. Nếu phát sinh nhiệm vụ mới, bổ sung vào task-board thay vì tự ý mở rộng phạm vi.

Khi phát hiện lỗi ngoài phạm vi:
- Ghi nhận lỗi.
- Thêm nhiệm vụ mới nếu cần.
- Không tự ý chiếm nhiệm vụ của Agent khác.

## 8. TASK STATUS UPDATE

Cập nhật trạng thái theo thời gian thực khi có thể.

### Khi bắt đầu

Chuyển:

`[ ] TODO` → `[~] IN_PROGRESS`

Ghi Agent ID và thời gian bắt đầu.

### Khi gặp trở ngại

Chuyển:

`[~] IN_PROGRESS` → `[!] BLOCKED`

Ghi rõ:
- Nguyên nhân.
- Vấn đề cần giải quyết.
- Nhiệm vụ hoặc Agent liên quan.

### Khi hoàn thành

Chỉ chuyển:

`[~] IN_PROGRESS` → `[x] DONE`

Sau khi:
- Code đã được thực hiện.
- Kiểm tra hoặc test phù hợp đã chạy.
- Không còn lỗi nghiêm trọng chưa xử lý trong phạm vi nhiệm vụ.
- Đã ghi lại file thay đổi.
- Đã cập nhật context liên quan.

Không đánh dấu DONE chỉ vì đã viết xong code.

## 9. DEPENDENCY MANAGEMENT

Nếu nhiệm vụ phụ thuộc vào nhiệm vụ khác:

- Ghi ID nhiệm vụ phụ thuộc.
- Không thực hiện phần cần đầu vào chưa sẵn sàng.
- Có thể thực hiện những phần độc lập trước.
- Theo dõi trạng thái nhiệm vụ phụ thuộc.
- Tiếp tục khi điều kiện đã được đáp ứng.

Không tự ý thay đổi giao diện API hoặc cấu trúc dữ liệu chung để né tránh dependency.

## 10. SHARED CONTEXT UPDATE

Sau khi hoàn thành nhiệm vụ:

Cập nhật các file cần thiết:

- `context/progress.md`
- `context/architecture.md`
- `context/decisions.md`
- `context/issues.md`
- `context/changelog.md`
- `context/sessions/YYYY-MM-DD.md`

Đồng thời cập nhật `context/task-board.md`.

Chỉ ghi nhận thông tin đã thực hiện hoặc kiểm chứng.

Không xóa nội dung lịch sử của Agent khác.

## 11. COMMUNICATION BETWEEN AGENTS

Mọi thông tin quan trọng cần được lưu trong file dùng chung.

Không giả định Agent khác có thể đọc lịch sử hội thoại riêng của bạn.

Khi cần phối hợp, ghi rõ:

- Agent gửi.
- Agent nhận.
- Nội dung cần phối hợp.
- File hoặc module liên quan.
- Nhiệm vụ liên quan.
- Trạng thái phản hồi.

Không coi việc ghi vào task-board là một cơ chế gửi thông báo trực tiếp cho Agent khác.

Nếu không có hệ thống giao tiếp trực tiếp, sử dụng file chung làm nguồn thông tin để các Agent kiểm tra khi bắt đầu hoặc tiếp tục công việc.

## 12. COMPLETION REPORT

Khi kết thúc nhiệm vụ, báo cáo:

### Task
- ID:
- Tên:
- Agent:
- Trạng thái:

### Implementation
- Nội dung đã thực hiện.
- File đã tạo.
- File đã chỉnh sửa.

### Verification
- Các lệnh test.
- Kết quả kiểm tra.
- Lỗi còn tồn tại.

### Context
- Các file context đã cập nhật.
- Những quyết định mới.
- Những vấn đề cần Agent khác xử lý.

### Next Tasks
- Nhiệm vụ tiếp theo.
- Dependency còn tồn tại.
- Agent cần phối hợp.

## 13. ABSOLUTE RULES

1. Luôn đọc context trước khi làm.
2. Luôn kiểm tra task-board trước khi nhận việc.
3. Luôn đăng ký nhiệm vụ trước khi sửa code.
4. Không nhận nhiệm vụ đã có Agent khác thực hiện.
5. Không tự ý chỉnh sửa phạm vi của Agent khác.
6. Không đánh dấu hoàn thành khi chưa kiểm chứng.
7. Không xóa thay đổi của Agent khác.
8. Không tạo nhiệm vụ trùng lặp.
9. Luôn cập nhật trạng thái và kết quả.
10. Ưu tiên bảo toàn tính nhất quán của toàn bộ dự án.

**CORE PRINCIPLE: Read Context → Claim Task → Execute → Verify → Update Shared Context → Report.**