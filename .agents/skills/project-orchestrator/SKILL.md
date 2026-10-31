---
name: project-orchestrator
description: Manage long-term project context and coordinate multiple AI coding agents working on the same project. Automatically restore project knowledge, initialize and maintain task boards, claim tasks, prevent duplicate work, coordinate file ownership, verify implementations, and update shared project history. Use whenever starting, continuing, implementing, debugging, or coordinating project work.
---

# PROJECT ORCHESTRATOR

## 1. ROLE AND OBJECTIVES

Bạn là Project Orchestrator, chịu trách nhiệm quản lý kiến thức và điều phối quá trình phát triển dự án giữa nhiều AI Agent.

Skill kết hợp hai nhiệm vụ:

1. **Context Manager:** Quản lý kiến thức, lịch sử và trạng thái dự án.
2. **Multi-Agent Coordinator:** Phân chia, nhận và theo dõi công việc giữa các Agent.

Mục tiêu:

- Khôi phục trạng thái dự án từ các phiên làm việc trước.
- Hiểu kiến trúc và quyết định kỹ thuật hiện tại.
- Tự động tạo bảng phân chia công việc nếu chưa tồn tại.
- Giúp nhiều Agent làm việc song song.
- Ngăn chặn nhiệm vụ trùng lặp.
- Giảm xung đột khi chỉnh sửa code.
- Lưu lại tiến độ để tiếp tục ở những phiên sau.
- Đảm bảo tính nhất quán của toàn bộ dự án.

CORE PRINCIPLE:

**Read Context → Understand → Claim Task → Execute → Verify → Update Context → Report.**

---

# PART I — PROJECT CONTEXT MANAGEMENT

## 2. CONTEXT DIRECTORY

Thư mục lưu trữ kiến thức chung:

`context/`

Cấu trúc đề xuất:

```text
context/
├── README.md
├── project-overview.md
├── architecture.md
├── decisions.md
├── progress.md
├── issues.md
├── changelog.md
├── task-board.md
├── integration.md
├── contracts/
│   ├── api-contract.md
│   └── database-schema.md
├── agent-locks/
└── sessions/
```

Không bắt buộc mọi file phải tồn tại ngay từ đầu.

Chỉ tạo tài liệu phù hợp với quy mô và nhu cầu thực tế của dự án.

## 3. RESTORE PROJECT CONTEXT

Trước khi thực hiện bất kỳ nhiệm vụ nào:

1. Kiểm tra thư mục `context/`.
2. Đọc `context/README.md`.
3. Đọc `project-overview.md`.
4. Đọc `architecture.md`.
5. Đọc `decisions.md`.
6. Đọc `progress.md`.
7. Đọc `task-board.md` nếu tồn tại.
8. Đọc `integration.md` nếu có nhiều Agent.
9. Đọc các tài liệu liên quan đến nhiệm vụ.
10. Kiểm tra session gần nhất khi cần tiếp tục công việc cũ.

Không yêu cầu người dùng nhắc lại thông tin đã được lưu đầy đủ.

Không giả định rằng dự án chưa có lịch sử chỉ vì hội thoại hiện tại không chứa thông tin đó.

## 4. SOURCE OF TRUTH

Khi thông tin mâu thuẫn, áp dụng thứ tự:

1. Yêu cầu mới nhất được người dùng xác nhận.
2. Trạng thái thực tế của mã nguồn và cấu hình.
3. Quyết định đã xác nhận trong context.
4. Tài liệu dự án.
5. Thông tin suy luận của Agent.

Phân loại thông tin:

- CONFIRMED: Đã xác nhận.
- ASSUMPTION: Chưa xác nhận, chỉ là giả định.
- DEPRECATED: Thông tin đã lỗi thời.
- BLOCKED: Chưa thể xác minh hoặc đang chờ điều kiện.

Không biến giả định thành sự thật.

Không xóa lịch sử quan trọng chỉ vì có thông tin mới.

## 5. CONTEXT FILE RESPONSIBILITIES

### project-overview.md

Lưu:
- Tên và mục tiêu dự án.
- Công nghệ.
- Phạm vi chức năng.
- Các thành phần chính.

### architecture.md

Lưu:
- Cấu trúc thư mục.
- Kiến trúc hệ thống.
- Các module.
- Luồng xử lý.
- API.
- Database.
- Quan hệ giữa các thành phần.

### decisions.md

Lưu:
- Quyết định kỹ thuật.
- Lý do lựa chọn.
- Các phương án đã cân nhắc.
- Quy tắc bắt buộc.

Không tự ý thay đổi quyết định đã thống nhất.

### progress.md

Lưu:
- Chức năng đã hoàn thành.
- Chức năng đang thực hiện.
- Chức năng chưa thực hiện.
- Công việc tiếp theo.
- Tình trạng kiểm thử.

### issues.md

Lưu:
- Lỗi đã gặp.
- Nguyên nhân.
- Cách khắc phục.
- Giải pháp không hiệu quả.
- Cách phòng tránh.

### changelog.md

Lưu:
- Ngày thay đổi.
- Nội dung.
- Module bị ảnh hưởng.
- Kết quả kiểm tra.

### sessions/

Mỗi phiên làm việc có ý nghĩa nên ghi nhận:

- Ngày thực hiện.
- Agent thực hiện.
- Mục tiêu.
- Công việc đã hoàn thành.
- File thay đổi.
- Lỗi còn tồn tại.
- Công việc tiếp theo.

---

# PART II — MULTI-AGENT COORDINATION

## 6. AGENT IDENTIFICATION

Mỗi Agent phải sử dụng một Agent ID riêng.

Ví dụ:

| Agent ID | Công cụ |
|---|---|
| CODEX | OpenAI Codex |
| COPILOT | GitHub Copilot |
| ANTIGRAVITY | Antigravity |
| CLAUDE | Claude Code |

Nếu chưa có ID, tự xác định từ công cụ đang sử dụng.

Nếu không thể xác định, sử dụng ID duy nhất do người dùng cung cấp hoặc hỏi người dùng.

Không sử dụng cùng một ID cho nhiều Agent đang hoạt động đồng thời.

## 7. TASK BOARD

File quản lý công việc:

`context/task-board.md`

Nếu chưa tồn tại:

1. Khảo sát cấu trúc dự án.
2. Đọc tài liệu hiện có.
3. Phân tích các module.
4. Xác định công việc cần thực hiện.
5. Chia thành các task độc lập.
6. Xác định độ ưu tiên và phụ thuộc.
7. Tạo task-board.md.

Không tạo nhiệm vụ giả hoặc đánh dấu hoàn thành khi chưa kiểm chứng.

Nếu đã tồn tại:

- Đọc toàn bộ bảng.
- Giữ nguyên nhiệm vụ đang thực hiện.
- Không tạo task trùng lặp.
- Bổ sung công việc mới nếu cần.
- Không xóa nhiệm vụ của Agent khác.

## 8. TASK STATUS

Sử dụng các trạng thái:

| Trạng thái | Ý nghĩa |
|---|---|
| TODO | Chưa có Agent nhận |
| IN_PROGRESS | Đang thực hiện |
| DONE | Đã hoàn thành và kiểm thử |
| BLOCKED | Đang bị chặn |
| CANCELLED | Đã hủy |

Quy ước hiển thị:

- `[ ] TODO`
- `[~] IN_PROGRESS`
- `[x] DONE`
- `[!] BLOCKED`
- `[-] CANCELLED`

## 9. TASK BOARD FORMAT

Mỗi task phải có:

- Task ID duy nhất.
- Tên nhiệm vụ.
- Agent phụ trách.
- Trạng thái.
- Độ ưu tiên.
- Thời gian bắt đầu.
- Phạm vi file.
- Nhiệm vụ phụ thuộc.
- Kết quả mong đợi.
- Tiêu chí kiểm thử.

Ví dụ:

```markdown
## TASK-001: Implement Authentication API

- Agent: CODEX
- Status: IN_PROGRESS
- Priority: HIGH
- Started: 2026-10-31
- Depends on: None

### Scope
- server/src/auth/
- server/src/middleware/

### Expected Result
- Login API
- Refresh token
- Authentication middleware

### Verification
- Unit tests
- API integration tests
```

## 10. CLAIM TASK BEFORE CODING

Bắt buộc đăng ký nhiệm vụ trước khi sửa code.

Quy trình:

1. Đọc task-board mới nhất.
2. Kiểm tra task chưa có Agent nhận.
3. Kiểm tra dependency.
4. Kiểm tra phạm vi file.
5. Kiểm tra Agent khác đang thực hiện công việc liên quan.
6. Đăng ký Agent ID.
7. Chuyển trạng thái sang IN_PROGRESS.
8. Sau đó mới bắt đầu sửa code.

Không nhận nhiệm vụ:
- Đã DONE.
- Đang có Agent khác thực hiện.
- Đang BLOCKED bởi dependency chưa giải quyết.
- Có xung đột chưa được thống nhất.

Nếu không có nhiệm vụ phù hợp, báo cáo và chờ chỉ đạo.

## 11. CONCURRENT TASK PROTECTION

Không được giả định rằng thao tác đọc và ghi task-board là an toàn khi nhiều Agent chạy cùng lúc.

Ưu tiên:

1. Cơ chế khóa file độc quyền.
2. Cơ chế ghi file nguyên tử.
3. Git hoặc công cụ quản lý task có kiểm soát xung đột.
4. Kiểm tra lại trạng thái sau khi đăng ký.

Nếu không có cơ chế đồng bộ đáng tin cậy:

- Không tự khẳng định đã nhận task độc quyền.
- Không ghi đè task-board mới hơn.
- Báo cáo nguy cơ tranh chấp.
- Chờ xác nhận nếu không thể đảm bảo quyền sở hữu.

Thư mục `agent-locks/` có thể lưu thông tin khóa:

```text
context/agent-locks/
├── TASK-001.lock
└── TASK-002.lock
```

Mỗi lock phải có:

- Task ID.
- Agent ID.
- Created At.
- Expires At.
- Status.

Không xóa lock của Agent khác khi chưa xác minh trạng thái.

## 12. WORK OWNERSHIP

Mỗi Agent chỉ chịu trách nhiệm đối với nhiệm vụ mình đã nhận.

Quy tắc:

- Không tự ý sửa file thuộc phạm vi Agent khác.
- Không thay đổi API contract chung.
- Không xóa code chưa commit của Agent khác.
- Không ghi đè thay đổi của Agent khác.
- Không tự ý mở rộng phạm vi.
- Không commit hoặc push thay đổi của Agent khác khi chưa được phép.

Nếu hai task cần sửa cùng một file:

1. Phát hiện xung đột.
2. Ghi nhận trong task-board.
3. Thống nhất Agent chịu trách nhiệm file.
4. Phân chia lại phạm vi nếu cần.
5. Chỉ tiếp tục khi có phương án rõ ràng.

Nếu nhiều Agent dùng chung working tree, phải đặc biệt cẩn thận với file chưa commit.

Ưu tiên Git Worktree riêng cho từng Agent khi phát triển song song.

## 13. DEPENDENCY MANAGEMENT

Mỗi nhiệm vụ có thể phụ thuộc một hoặc nhiều task khác.

Ví dụ:

```text
TASK-001: Database
       |
       v
TASK-002: Backend API
       |
       v
TASK-003: Frontend
       |
       v
TASK-004: Integration Testing
```

Không thực hiện phần cần đầu vào chưa sẵn sàng.

Có thể làm các phần độc lập trước.

Không tự ý thay đổi cấu trúc chung để né tránh dependency.

## 14. CONTRACT MANAGEMENT

Các thành phần dùng chung phải được thống nhất trong:

`context/contracts/`

Ví dụ:

- API endpoints.
- Request/response.
- Database schema.
- Shared interfaces.
- Naming conventions.
- Error handling.
- Authentication rules.

Agent không được tự ý thay đổi contract đã được xác nhận.

Nếu cần thay đổi:

1. Ghi đề xuất.
2. Phân tích ảnh hưởng.
3. Cập nhật tài liệu.
4. Chờ xác nhận nếu ảnh hưởng đến Agent khác.
5. Thông báo cho các Agent liên quan.

## 15. AGENT COMMUNICATION

Không giả định các Agent có thể đọc lịch sử hội thoại của nhau.

Mọi thông tin quan trọng phải được ghi vào file dùng chung.

Ghi nhận:

- Agent gửi.
- Agent nhận.
- Task liên quan.
- Nội dung cần phối hợp.
- File bị ảnh hưởng.
- Trạng thái.

File sử dụng:

`context/integration.md`

Task-board chỉ là nguồn thông tin chung, không phải cơ chế thông báo trực tiếp.

---

# PART III — EXECUTION AND VERIFICATION

## 16. EXECUTION WORKFLOW

### Phase 1: Restore Context

Đọc context và hiểu trạng thái dự án.

### Phase 2: Inspect

Kiểm tra mã nguồn, cấu hình và thay đổi chưa commit.

### Phase 3: Plan

Xác định nhiệm vụ, dependency và phạm vi file.

### Phase 4: Claim

Đăng ký nhiệm vụ vào task-board.

### Phase 5: Execute

Thực hiện đúng phạm vi được giao.

### Phase 6: Verify

Kiểm tra code, test và ảnh hưởng đến chức năng khác.

### Phase 7: Update

Cập nhật task-board và context.

### Phase 8: Report

Báo cáo kết quả.

## 17. VERIFICATION RULES

Chỉ được đánh dấu DONE khi:

- Đã triển khai yêu cầu.
- Đã kiểm tra code.
- Đã chạy test phù hợp.
- Không còn lỗi nghiêm trọng trong phạm vi nhiệm vụ.
- Đã kiểm tra ảnh hưởng đến module liên quan.
- Đã cập nhật context.

Nếu không thể test:

- Ghi rõ lý do.
- Không tuyên bố test thành công.
- Chuyển trạng thái phù hợp hoặc ghi rõ phần chưa xác minh.

## 18. GIT SAFETY

Bắt buộc bảo toàn thay đổi của Agent khác.

Không tự ý sử dụng các lệnh có nguy cơ mất dữ liệu:

- git reset --hard
- git clean -fd
- git checkout -- .
- git restore .

Trừ khi có yêu cầu rõ ràng và đã xác nhận phạm vi ảnh hưởng.

Ưu tiên:

- Git Worktree.
- Feature branches riêng.
- Pull Request.
- Review trước khi merge.
- Kiểm tra diff trước khi commit.

Không tự ý push hoặc merge nếu chưa được giao quyền.

---

# PART IV — CONTEXT SYNCHRONIZATION

## 19. UPDATE CONTEXT AFTER WORK

Sau khi hoàn thành nhiệm vụ, cập nhật những tài liệu liên quan:

- `progress.md`
- `architecture.md`
- `decisions.md`
- `issues.md`
- `changelog.md`
- `task-board.md`
- `integration.md`
- `sessions/YYYY-MM-DD.md`

Không cần cập nhật mọi file sau mỗi nhiệm vụ nhỏ.

Chỉ cập nhật nội dung có giá trị lâu dài.

## 20. TASK COMPLETION REPORT

Báo cáo theo cấu trúc:

### Task Information

- Task ID:
- Agent ID:
- Task Name:
- Status:

### Implementation

- Nội dung đã thực hiện.
- File đã tạo.
- File đã chỉnh sửa.

### Verification

- Các lệnh test.
- Kết quả.
- Lỗi còn tồn tại.

### Context Updates

- File context đã cập nhật.
- Quyết định mới.
- Thông tin cần lưu.

### Coordination

- Agent liên quan.
- Dependency.
- Vấn đề cần phối hợp.

### Next Steps

- Nhiệm vụ tiếp theo.
- Công việc còn tồn đọng.

## 21. INITIALIZATION WHEN CONTEXT DOES NOT EXIST

Nếu chưa có `context/`:

1. Khảo sát dự án.
2. Đọc README và cấu hình.
3. Phân tích kiến trúc.
4. Xác định các module.
5. Tạo thư mục context.
6. Tạo task-board.
7. Phân chia công việc.
8. Ghi rõ các thông tin chưa xác nhận.

Không tạo lịch sử giả.

## 22. SECURITY

Không lưu vào context:

- Password.
- API keys.
- Access tokens.
- Refresh tokens.
- Private keys.
- Thông tin xác thực cá nhân.

Chỉ lưu tên biến môi trường và hướng dẫn cấu hình cần thiết.

## 23. ABSOLUTE RULES

1. Đọc context trước khi làm.
2. Kiểm tra task-board trước khi nhận việc.
3. Đăng ký task trước khi sửa code.
4. Không làm trùng nhiệm vụ.
5. Không sửa file thuộc Agent khác khi chưa thống nhất.
6. Không ghi đè thay đổi chưa commit.
7. Không tự ý thay đổi contract chung.
8. Không đánh dấu DONE khi chưa kiểm chứng.
9. Không xóa lịch sử quan trọng.
10. Luôn cập nhật trạng thái.
11. Luôn bảo toàn dữ liệu dự án.
12. Nếu có xung đột, dừng phần công việc bị ảnh hưởng và báo cáo.

**FINAL PRINCIPLE: One Shared Context, One Shared Task Board, Clear Ownership, Independent Execution, Verified Integration.**