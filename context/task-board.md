# Task Board — Phân loại chất lượng và độ chín trái cây

> Cập nhật lần cuối: 2026-10-31  
> Tuân thủ skill: `multi-agent-coordinator`

---

## Quy ước trạng thái

| Ký hiệu | Trạng thái | Ý nghĩa |
|---|---|---|
| `[ ]` | TODO | Chưa có Agent nhận |
| `[~]` | IN_PROGRESS | Đang thực hiện |
| `[x]` | DONE | Hoàn thành, đã kiểm tra |
| `[!]` | BLOCKED | Đang bị chặn bởi dependency |
| `[-]` | CANCELLED | Đã hủy |

---

## 🔴 NHÓM A — Cơ sở hạ tầng & Dataset (Đã hoàn thành 100%)

| ID | Nhiệm vụ | Agent | Trạng thái | Ghi chú |
|---|---|---|---|---|
| TASK-A01 | Khảo sát repository, thống nhất yêu cầu, thiết lập môi trường Python 3.14 + uv + venv | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-A02 | Cập nhật `requirements.txt` (bỏ torch/torchvision, thêm scikit-learn, scikit-image, scipy) | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-A03 | Viết `src/dataset.py` — Parser YOLO + crop bbox + resize 128×128 | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-A04 | Sửa lỗi làm tròn `_parse_yolo_box` (`int` → `int(round())`) | Main Agent | `[x] DONE` | Phiên 2026-10-30 |

---

## 🟠 NHÓM B — ML Pipeline (Đã hoàn thành 100%)

| ID | Nhiệm vụ | Agent | Trạng thái | Ghi chú |
|---|---|---|---|---|
| TASK-B01 | Viết `src/features.py` — 51 đặc trưng (Color, LBP, GLCM, Shape) | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-B02 | Viết `src/train.py` — Pipeline 7 mô hình + GridSearch + lưu artifact | Main Agent | `[x] DONE` | Hoàn thành |
| TASK-B03 | Viết `src/evaluate.py` — Metrics + 4 biểu đồ báo cáo | Main Agent | `[x] DONE` | Hoàn thành |
| TASK-B04 | **Chạy huấn luyện thực tế** `python -m src.train` và sinh `models/` + `reports/` | Main Agent | `[x] DONE` | SVM Test Acc: 93.50%, F1: 93.71% |

---

## 🟡 NHÓM C — Inference & Giao diện (Đã hoàn thành 100%)

| ID | Nhiệm vụ | Agent | Trạng thái | Ghi chú |
|---|---|---|---|---|
| TASK-C01 | Viết `src/predict.py` — FruitClassifier inference engine | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-C02 | Viết `app/streamlit_app.py` — Web UI đầy đủ 3 tab | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-C03 | Unit tests `tests/test_pipeline.py` — 17/17 passed | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-C04 | **Kiểm thử Streamlit end-to-end** với ảnh thực tế và probe máy chủ | Main Agent | `[x] DONE` | Suy luận 10.7ms, HTTP 200 OK |

---

## 🟢 NHÓM D — Tài liệu & Báo cáo (Đã hoàn thành 100%)

| ID | Nhiệm vụ | Agent | Trạng thái | Ghi chú |
|---|---|---|---|---|
| TASK-D01 | Tạo hệ thống `context/` theo skill `context-manager` | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-D02 | **Cập nhật `README.md`** với lệnh uv, cấu hình và cách chạy | Main Agent | `[x] DONE` | Hoàn thành 2026-10-31 |
| TASK-D03 | Viết `docs/` — Hướng dẫn cài đặt, training, inference, kết quả thực nghiệm | Main Agent | `[x] DONE` | Đầy đủ 4 tài liệu trong `docs/` |
| TASK-D04 | Cập nhật `data/README.md` — Phản ánh cấu trúc dataset YOLO thực tế | Main Agent | `[x] DONE` | Đã rà soát khớp 100% |

---

## 🔵 NHÓM E — Kiểm tra chất lượng & Hoàn thiện (Đã hoàn thành 100%)

| ID | Nhiệm vụ | Agent | Trạng thái | Ghi chú |
|---|---|---|---|---|
| TASK-E01 | Chạy `pytest` cuối cùng để xác nhận 17/17 passed sau mọi thay đổi | Main Agent | `[x] DONE` | 17/17 passed in 1.82s |
| TASK-E02 | Kiểm tra biên dịch mã nguồn toàn bộ `src/`, `app/`, `tests/` | Main Agent | `[x] DONE` | py_compile thành công 100% |
| TASK-E03 | Lập báo cáo tiến độ tổng kết và cập nhật toàn bộ context | Main Agent | `[x] DONE` | Hoàn thành 2026-10-31 |

## 🟣 NHÓM F — Demo giao diện

| ID | Nhiệm vụ | Agent | Trạng thái | Ghi chú |
|---|---|---|---|---|
| TASK-F01 | Thiết kế lại chương trình demo Streamlit theo frontend-design | `Copilot Session 0bc66139` | `[x] DONE` | Hoàn thành 2026-10-31 13:23 · Giao diện tiếng Việt mới; kiểm thử upload, dự đoán, lịch sử, thông tin mô hình và viewport mobile |

## 🟠 NHÓM G — Sửa lỗi chạy demo

| ID | Nhiệm vụ | Agent | Trạng thái | Ghi chú |
|---|---|---|---|---|
| TASK-G01 | Sửa import `src` và đường dẫn tài nguyên khi chạy Streamlit từ thư mục khác | `Copilot Session 0bc66139` | `[x] DONE` | Hoàn thành 2026-10-31 13:30 · Đã xác thực từ `app/`; 17/17 test đạt |

## 🟢 NHÓM H — Chuyển đổi Web Application

### TASK-H01: Thay giao diện Streamlit bằng Web Application HTML/CSS/JavaScript

- Agent: `CODEX`
- Status: DONE
- Priority: HIGH
- Started: 2026-10-31
- Depends on: TASK-C01 (FruitClassifier), TASK-C02 (giao diện hiện có làm tài liệu tham chiếu); cả hai đã DONE

#### Scope
- `app/web/`
- `requirements.txt`
- `README.md`
- `docs/inference.md`
- `tests/test_web_api.py`
- Các tài liệu liên quan trong `context/`

#### Expected Result
- FastAPI dùng `src/predict.py` để cung cấp health, prediction, model info, history và statistics.
- Giao diện HTML/CSS/JavaScript responsive với các màn hình dashboard, nhận diện, lịch sử, thống kê và thông tin mô hình.
- Lịch sử lưu metadata dự đoán thật trong SQLite; không lưu ảnh hoặc tạo số liệu minh họa.
- Giữ nguyên `app/streamlit_app.py`, `src/`, dataset và model artifacts.

#### Verification
- Kiểm thử API bằng ảnh/bytes tổng hợp hoặc dependency injection; không yêu cầu dataset.
- Chạy lại `pytest tests/` hiện có và kiểm tra frontend/API phục vụ qua HTTP nếu môi trường hỗ trợ.
- Biên dịch Python và rà `git diff --check`.
- Kết quả: `24 passed`; Python compile và JavaScript syntax đều đạt; OpenAPI snapshot khớp.
- Smoke test model SVM thật: `Mango Overripe`, confidence 84,18%, 9 xác suất cộng thành 1; chỉ ghi vào SQLite tạm rồi xóa.
- HTTP smoke: `/api/health`, `/`, trang nhận diện, CSS và `/openapi.json` trả HTTP 200.
- Postman CLI không khởi động vì policy manifest của môi trường; thay thế bằng FastAPI TestClient và HTTP smoke.

---

## Dependency Graph

```
TASK-A01,A02,A03,A04 (Done)
    ↓
TASK-B01,B02,B03,B04 (Done)
    ↓
TASK-C01,C02,C03,C04 (Done)
    ↓
TASK-D01,D02,D03,D04 (Done)
    ↓
TASK-E01,E02,E03 (Done)
    ↓
TASK-F01 [DONE]
    ↓
TASK-G01 [DONE]
    ↓
TASK-H01 [DONE]
    ↓
TASK-I01 [IN_PROGRESS]
```

---

## Agent Registry

| Agent ID | Vai trò | Phiên làm việc | Nhiệm vụ hiện tại |
|---|---|---|---|
| `Main Agent` | Primary Developer | 2026-10-30 → nay | Không có task đang chạy |
| `Copilot Session 0bc66139` | Copilot SDK session | 2026-10-31 | Hoàn thành TASK-G01 |
| `CODEX` | OpenAI Codex | 2026-10-31 | Đang thực hiện TASK-I01 |

---

## Nhật ký cập nhật

| Thời gian | Hành động | Agent |
|---|---|---|
| 2026-10-31 12:40 | Tạo `task-board.md` từ khảo sát context và tiến độ | Main Agent |
| 2026-10-31 12:40 | Đăng ký TASK-D02 `IN_PROGRESS` | Main Agent |
| 2026-10-31 12:46 | Hoàn thành TASK-D02 và rà soát hoàn tất TASK-D04 | Main Agent |
| 2026-10-31 12:47 | Đăng ký TASK-B04 `IN_PROGRESS` | Main Agent |
| 2026-10-31 12:59 | Hoàn thành TASK-B04: Huấn luyện 7 mô hình, lưu checkpoint `models/` và biểu đồ `reports/` | Main Agent |
| 2026-10-31 13:02 | Hoàn thành TASK-C04: Kiểm thử suy luận và web Streamlit HTTP 200 | Main Agent |
| 2026-10-31 13:03 | Hoàn thành TASK-D03: Viết 4 tài liệu kỹ thuật trong `docs/` | Main Agent |
| 2026-10-31 13:04 | Hoàn thành TASK-E01, E02, E03: Kiểm thử 17/17 tests passed, cập nhật context hoàn tất | Main Agent |
| 2026-10-31 13:17 | Đăng ký TASK-F01 `IN_PROGRESS` — Thiết kế lại demo Streamlit theo frontend-design | Copilot Session 0bc66139 |
| 2026-10-31 13:23 | Hoàn thành TASK-F01: demo Streamlit mới, kiểm tra suy luận ảnh thật và responsive mobile | Copilot Session 0bc66139 |
| 2026-10-31 13:29 | Đăng ký TASK-G01 `IN_PROGRESS` — Sửa lỗi chạy Streamlit ngoài thư mục gốc | Copilot Session 0bc66139 |
| 2026-10-31 13:30 | Hoàn thành TASK-G01: xác thực import và nạp model khi chạy từ `app/` | Copilot Session 0bc66139 |
| 2026-10-31 13:38 | Đăng ký TASK-H01 `IN_PROGRESS` — Thay UI Streamlit bằng FastAPI và HTML/CSS/JavaScript | CODEX |
| 2026-10-31 13:54 | Hoàn thành TASK-H01: FastAPI, frontend HTML/CSS/JavaScript, OpenAPI, tài liệu; 24/24 tests và HTTP smoke đạt | CODEX |

## 🔵 NHÓM I — Huấn luyện tùy chọn trên GPU

### TASK-I01: Thêm backend PyTorch CUDA tùy chọn cho pipeline huấn luyện

- Agent: `CODEX`
- Status: IN_PROGRESS
- Priority: MEDIUM
- Started: 2026-10-31
- Scope: `src/train.py`, `src/torch_model.py`, `requirements.txt`, `docs/`, `README.md`, `context/`
- Depends on: TASK-B02, TASK-C01

### Expected Result
- Giữ pipeline scikit-learn làm mặc định; thêm lựa chọn PyTorch chạy trên CUDA.
- PyTorch được khai báo trong `requirements.txt`; GPU vẫn chỉ được dùng khi chọn `--device cuda`.
- Artifact GPU có thể được `FruitClassifier` nạp để suy luận.

### Verification
- Đã hoàn thiện thay đổi mã/tài liệu/context; chưa chạy test/runtime trong phiên này.
- Cần xác nhận bằng PyTorch CUDA trên máy có GPU NVIDIA trước khi đánh dấu DONE.

### Activity
- 2026-10-31: Đăng ký TASK-I01 `IN_PROGRESS` — Thêm backend huấn luyện PyTorch CUDA tùy chọn.
- 2026-10-31: Theo yêu cầu người dùng, thêm PyTorch vào `requirements.txt`; wheel cung cấp CUDA runtime nhưng không cài NVIDIA driver.
- 2026-10-31: Khóa PyTorch 2.14.1 + CUDA 12.6 sau khi đối chiếu RTX 3050/driver 566.07 và wheel Windows/Python 3.14.
