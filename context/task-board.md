# Task board

## Gần nhất

| ID | Task | Trạng thái | Kết quả |
|---|---|---|---|
| TASK-V01 | Lưu và hiển thị biểu đồ thuật toán | DONE | Web chọn đúng thư mục báo cáo theo `dataset_mode`; 11 test web đạt |
| TASK-O01 | Sửa GridSearch SVM calibration | DONE | Clone estimator, tham số `estimator__*`, regression test |
| TASK-P01 | Whole-scene mặc định, train CPU | DONE | Bỏ GPU/TorchMLP khỏi CLI và requirements |
| TASK-Q01 | Rút gọn context | DONE | Tóm tắt trạng thái hiện tại, bỏ lịch sử obsolete |
| TASK-D02 | Cập nhật số lượng dataset trong tài liệu | DONE | Cập nhật số lượng theo 9 lớp; đánh dấu kết quả cũ chưa được xác minh trên split mới |

## Kiểm thử TASK-P01

- `python -m pytest tests -q`: 32 passed; 1 cảnh báo Starlette/httpx.
- `py_compile`, CLI help và `git diff --check` đạt.
- Không chạy train trên dataset thật.

## Kiểm thử TASK-V01

- `python -m pytest tests\test_web_api.py -q`: 11 passed; 1 cảnh báo Starlette/httpx.
- `python -m pytest tests -q`: 33 passed, 1 failed. Test train kỳ vọng mặc định
  `whole-scene`, nhưng thay đổi chưa commit trong `src/train.py` hiện đặt mặc định
  `folder`; không sửa thay đổi này trong task biểu đồ.
- `py_compile` và `git diff --check` đạt.
