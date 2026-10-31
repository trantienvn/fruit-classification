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

## 🔴 NHÓM A — Cơ sở hạ tầng & Dataset (Đã hoàn thành)

| ID | Nhiệm vụ | Agent | Trạng thái | Ghi chú |
|---|---|---|---|---|
| TASK-A01 | Khảo sát repository, thống nhất yêu cầu, thiết lập môi trường Python 3.14 + uv + venv | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-A02 | Cập nhật `requirements.txt` (bỏ torch/torchvision, thêm scikit-learn, scikit-image, scipy) | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-A03 | Viết `src/dataset.py` — Parser YOLO + crop bbox + resize 128×128 | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-A04 | Sửa lỗi làm tròn `_parse_yolo_box` (`int` → `int(round())`) | Main Agent | `[x] DONE` | Phiên 2026-10-30 |

---

## 🟠 NHÓM B — ML Pipeline (Đã hoàn thành code, chờ thực thi)

| ID | Nhiệm vụ | Agent | Trạng thái | Ghi chú |
|---|---|---|---|---|
| TASK-B01 | Viết `src/features.py` — 51 đặc trưng (Color, LBP, GLCM, Shape) | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-B02 | Viết `src/train.py` — Pipeline 7 mô hình + GridSearch + lưu artifact | Main Agent | `[x] DONE` | Code sẵn sàng |
| TASK-B03 | Viết `src/evaluate.py` — Metrics + 4 biểu đồ báo cáo | Main Agent | `[x] DONE` | Code sẵn sàng |
| TASK-B04 | **Chạy huấn luyện thực tế** `python -m src.train` và sinh `models/` + `reports/` | — | `[ ] TODO` | **Ưu tiên cao — Dependency cho TASK-C04, D02** |

---

## 🟡 NHÓM C — Inference & Giao diện (Đã hoàn thành code)

| ID | Nhiệm vụ | Agent | Trạng thái | Ghi chú |
|---|---|---|---|---|
| TASK-C01 | Viết `src/predict.py` — FruitClassifier inference engine | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-C02 | Viết `app/streamlit_app.py` — Web UI đầy đủ 3 tab | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-C03 | Unit tests `tests/test_pipeline.py` — 17/17 passed | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-C04 | **Kiểm thử Streamlit end-to-end** với ảnh thực tế sau khi có model | — | `[!] BLOCKED` | Chờ TASK-B04 hoàn thành |

---

## 🟢 NHÓM D — Tài liệu & Báo cáo

| ID | Nhiệm vụ | Agent | Trạng thái | Ghi chú |
|---|---|---|---|---|
| TASK-D01 | Tạo hệ thống `context/` theo skill `context-manager` | Main Agent | `[x] DONE` | Phiên 2026-10-30 |
| TASK-D02 | **Cập nhật `README.md`** với lệnh uv, số liệu thực tế sau khi train | — | `[~] IN_PROGRESS` | **Agent: Main Agent · Bắt đầu: 2026-10-31 12:40** · File: `README.md` |
| TASK-D03 | Viết `docs/` — Hướng dẫn cài đặt, training, inference, API, kết quả thực nghiệm | — | `[ ] TODO` | Chờ TASK-B04 để có số liệu thật |
| TASK-D04 | Cập nhật `data/README.md` — Phản ánh cấu trúc dataset YOLO thực tế | — | `[ ] TODO` | Độc lập |

---

## 🔵 NHÓM E — Kiểm tra chất lượng & Hoàn thiện

| ID | Nhiệm vụ | Agent | Trạng thái | Ghi chú |
|---|---|---|---|---|
| TASK-E01 | Chạy `pytest` cuối cùng để xác nhận 17/17 passed sau mọi thay đổi | — | `[ ] TODO` | Chạy sau khi hoàn tất D02, D03, D04 |
| TASK-E02 | Kiểm tra lint và code style toàn bộ `src/` | — | `[ ] TODO` | Độc lập |
| TASK-E03 | Lập báo cáo tiến độ tổng kết cho đồ án học phần | — | `[ ] TODO` | Chờ B04, C04 |

---

## Dependency Graph

```
TASK-A01,A02,A03,A04 (Done)
    ↓
TASK-B01,B02,B03 (Done)
    ↓
TASK-B04 [TODO — Ưu tiên cao nhất]
    ↓
TASK-C04 [BLOCKED] ─────────────────→ TASK-E03
TASK-D02 [IN_PROGRESS] ──────────────→ TASK-E01
TASK-D03 [TODO, chờ B04]
TASK-D04 [TODO, độc lập]
```

---

## Agent Registry

| Agent ID | Vai trò | Phiên làm việc | Nhiệm vụ hiện tại |
|---|---|---|---|
| `Main Agent` | Primary Developer | 2026-10-30 → nay | TASK-D02 |

---

## Nhật ký cập nhật

| Thời gian | Hành động | Agent |
|---|---|---|
| 2026-10-31 12:40 | Tạo `task-board.md` từ khảo sát context và tiến độ | Main Agent |
| 2026-10-31 12:40 | Đăng ký TASK-D02 `IN_PROGRESS` | Main Agent |
