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
```

---

## Agent Registry

| Agent ID | Vai trò | Phiên làm việc | Nhiệm vụ hiện tại |
|---|---|---|---|
| `Main Agent` | Primary Developer | 2026-10-30 → nay | Không có task đang chạy |
| `Copilot Session 0bc66139` | Copilot SDK session | 2026-10-31 | Hoàn thành TASK-G01 |

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
