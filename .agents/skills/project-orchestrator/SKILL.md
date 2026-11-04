---
name: project-orchestrator
description: Quản lý context dự án và điều phối nhiều AI agent. Khôi phục kiến thức dự án, nhận/theo dõi task, tránh trùng việc và xung đột file, verify và cập nhật lịch sử chung. Dùng khi bắt đầu, tiếp tục, triển khai, debug hoặc phối hợp công việc dự án.
---

# PROJECT ORCHESTRATOR

Vòng lặp: **Đọc context → Nhận task → Làm → Verify → Cập nhật → Báo cáo.**
Ngắn gọn: chỉ ghi thông tin có giá trị lâu dài, không lặp nội dung giữa các file.

## 1. Chọn mức (ghi vào `context/README.md`)

| Mức | Khi nào | File cần có |
|---|---|---|
| Lite | 1 agent, dự án nhỏ/việc nhỏ | `README.md`, `tasks.md` |
| Standard | 1–2 agent, nhiều module | + `architecture.md`, `decisions.md`, `progress.md` |
| Full | ≥3 agent / hệ thống lớn | + `contracts/`, `locks/`, `integration.md`, `sessions/`, `issues.md`, `changelog.md` |

Chỉ nâng mức khi thật sự cần. Không tạo file rỗng.

## 2. Đọc context (lazy)

1. `context/README.md`: index ≤40 dòng (mục tiêu, stack, mức, danh sách file, Agent IDs).
2. `tasks.md`: chỉ phần liên quan + các task DOING.
3. Đọc thêm **chỉ khi task chạm tới**: `architecture.md` (module liên quan), `contracts/` (API/DB), `decisions.md` (khi chọn giải pháp), `integration.md` (khi có agent khác).
4. Không đọc toàn bộ changelog/sessions; chỉ session mới nhất khi tiếp tục việc cũ.

Không hỏi lại thông tin đã lưu. Không giả định dự án chưa có lịch sử.

**Chưa có `context/`:** quét nông (README, config, cây thư mục) → tạo `README.md` + `tasks.md` tối thiểu → đánh `ASSUMPTION` cho điều chưa xác minh. Không bịa lịch sử.

## 3. Nguồn sự thật

Ưu tiên khi mâu thuẫn: yêu cầu mới nhất của user > code/config thực tế > `decisions.md` > tài liệu > suy luận.
Nhãn: `CONFIRMED`, `ASSUMPTION`, `DEPRECATED`. Không biến giả định thành sự thật; thông tin cũ đánh DEPRECATED, không xóa.

## 4. Vai trò các file

- `README.md`: index + mục tiêu + stack.
- `architecture.md`: cấu trúc, module, luồng, quan hệ.
- `decisions.md`: quyết định + lý do + phương án loại. Không tự đổi quyết định đã chốt.
- `progress.md`: xong / đang làm / tiếp theo / tình trạng test.
- `issues.md`: lỗi, nguyên nhân, cách sửa, cách không hiệu quả.
- `changelog.md`: ngày, nội dung, module, kết quả test.
- `sessions/YYYY-MM-DD.md`: agent, mục tiêu, việc xong, file đổi, tồn đọng, bước tiếp.

## 5. Task board (`tasks.md`)

```markdown
| ID | Task | Agent | Status | Pri | Deps | Scope | Verify |
|---|---|---|---|---|---|---|---|
| T-001 | Auth API | CODEX | DOING | H | - | server/src/auth/ | unit+integration |
```

Status: `TODO` `DOING` `DONE` `BLOCKED` `CANCELLED`.
- ID duy nhất; task độc lập, scope file không chồng nhau.
- Task phức tạp (Full): chi tiết ở `tasks/T-001.md`.
- Chưa có board: tạo từ khảo sát thực tế. Có rồi: giữ nguyên task của agent khác, không tạo trùng, chỉ bổ sung.
- Không đánh DONE khi chưa kiểm chứng.

## 6. Nhận task (bắt buộc trước khi sửa code)

1. Đọc lại `tasks.md` ngay trước khi nhận.
2. Chọn task `TODO`, deps đã DONE, scope không chồng task DOING.
3. Ghi Agent ID + `DOING`.
   - Lite/Standard: sửa dòng task.
   - Full: tạo `locks/T-001.lock` bằng thao tác tạo file độc quyền (lỗi nếu đã tồn tại), gồm agent, created, expires.
4. Đọc lại để xác nhận mình là chủ, rồi mới code.

Không nhận task DONE / của agent khác / BLOCKED. Không có cơ chế đồng bộ đáng tin: không khẳng định độc quyền, báo nguy cơ tranh chấp và chờ xác nhận. Không xóa lock của agent khác khi chưa xác minh. Không có task phù hợp: báo cáo và chờ.

**Agent ID:** tự suy từ công cụ (CODEX, COPILOT, ANTIGRAVITY, CLAUDE). Không xác định/trùng: hỏi user.

## 7. Sở hữu & xung đột

- Chỉ làm trong scope đã nhận; không mở rộng scope, không sửa file của agent khác, không ghi đè thay đổi chưa commit của họ.
- Hai task cần cùng file: ghi vào `tasks.md`/`integration.md`, chốt agent chịu trách nhiệm hoặc chia lại scope, rồi mới tiếp tục.
- Có xung đột chưa giải quyết: dừng phần bị ảnh hưởng và báo cáo.
- Task phụ thuộc chưa xong: làm phần độc lập trước, không đổi cấu trúc chung để né dependency.
- Agent không đọc được hội thoại của nhau: thông tin phối hợp ghi vào `integration.md` (gửi, nhận, task, file, trạng thái).

## 8. Contracts (Full; Standard ghi trong `architecture.md`)

API, schema DB, interface dùng chung, naming, error handling, auth nằm ở `contracts/`. Không tự đổi contract đã chốt. Cần đổi: ghi đề xuất + ảnh hưởng, cập nhật tài liệu, chờ xác nhận nếu ảnh hưởng agent khác, thông báo qua `integration.md`.

## 9. Verify

DONE chỉ khi: đã triển khai đủ yêu cầu, đã chạy test phù hợp, không còn lỗi nghiêm trọng trong scope, đã kiểm tra module liên quan, đã cập nhật context.
Không test được: nêu lý do, không tuyên bố đã pass, ghi rõ phần chưa xác minh.

## 10. Git

Bảo toàn thay đổi của agent khác. Không dùng `git reset --hard`, `git clean -fd`, `git checkout -- .`, `git restore .` trừ khi user yêu cầu rõ. Ưu tiên worktree/branch riêng cho mỗi agent, xem diff trước khi commit. Không push/merge nếu chưa được giao quyền.

## 11. Cập nhật sau khi làm

Cập nhật `tasks.md` luôn; các file khác (`progress`, `architecture`, `decisions`, `issues`, `changelog`, `sessions`) chỉ khi có nội dung lâu dài. Task nhỏ không cần cập nhật hết.

## 12. Báo cáo (ngắn, bỏ mục rỗng)

```
Task: ID | Agent | Status
Đã làm: tóm tắt; file tạo/sửa
Verify: lệnh → kết quả; lỗi còn lại
Context: file đã cập nhật; quyết định mới
Phối hợp/Tiếp theo: agent liên quan, dependency, việc tồn đọng
```

## 13. Bảo mật

Không lưu vào context: password, API key, token, private key, thông tin xác thực cá nhân. Chỉ ghi tên biến môi trường và hướng dẫn cấu hình.

## 14. Quy tắc tuyệt đối

1. Đọc context liên quan trước khi làm.
2. Kiểm tra board và đăng ký task trước khi sửa code.
3. Không làm trùng; không sửa scope/file của agent khác khi chưa thống nhất.
4. Không ghi đè thay đổi chưa commit; không tự đổi contract/decision đã chốt.
5. Không DONE khi chưa kiểm chứng.
6. Không xóa lịch sử quan trọng; không bịa lịch sử.
7. Luôn cập nhật trạng thái task.
8. Xung đột: dừng phần bị ảnh hưởng và báo cáo.