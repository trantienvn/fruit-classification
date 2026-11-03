# Task board

Không có task đang thực hiện.

## Gần nhất

| ID | Task | Trạng thái | Kết quả |
|---|---|---|---|
| TASK-O01 | Sửa GridSearch SVM calibration | DONE | Clone estimator, tham số `estimator__*`, regression test |
| TASK-P01 | Whole-scene mặc định, train CPU | DONE | Bỏ GPU/TorchMLP khỏi CLI và requirements |
| TASK-Q01 | Rút gọn context | DONE | Tóm tắt trạng thái hiện tại, bỏ lịch sử obsolete |

## Kiểm thử TASK-P01

- `python -m pytest tests -q`: 32 passed; 1 cảnh báo Starlette/httpx.
- `py_compile`, CLI help và `git diff --check` đạt.
- Không chạy train trên dataset thật.
