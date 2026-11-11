# Task board

## Gần nhất

| ID | Task | Trạng thái | Kết quả |
|---|---|---|---|
| TASK-D03 | Cố định dataset train whole-scene | DONE | Bỏ CLI mode và metadata; 42 test đạt, CLI help và git diff --check đạt |
| TASK-R01 | Hoàn thiện pipeline ML theo yêu cầu phân loại độ chín | DONE | Code hoàn thiện, 42 test đạt; không chạy lại train theo chỉ đạo người dùng |
| TASK-V02 | Xác nhận Logistic Regression trong biểu đồ báo cáo | DONE | Kiểm thử huấn luyện LogisticRegression và sinh ảnh so sánh có nhãn thuật toán |
| TASK-V01 | Lưu và hiển thị biểu đồ thuật toán | DONE | Web chọn đúng thư mục báo cáo theo cấu hình model; 11 test web đạt |
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

## Kiểm thử TASK-V02

- Kiểm thử Logistic Regression tạo biểu đồ: 1 passed.
- `python -m pytest tests\test_pipeline.py -q -k "not
  training_cli_defaults_to_whole_scene_and_cpu"`: 21 passed.
- Test pipeline đầy đủ: 21 passed, 1 failed vì test đòi default
  `whole-scene` trong khi `src/train.py` hiện đặt `folder`; không liên quan
  đến Logistic Regression hay thay đổi task này.

## TASK-R01

- Agent: COPILOT
- Status: DONE
- Priority: HIGH
- Started: 2026-11-04
- Scope: `src/`, `tests/`, `docs/training.md`, `README.md`, context status files
- Expected: pipeline dùng taxonomy Cam/Chuối/Xoài × 3 trạng thái, preprocessing fold-local, báo cáo reproducible.
- Verification: `python -m pytest tests -q`: 42 passed, 1 warning Starlette; `python -m src.train --help` đạt; `git diff --check` đạt.
- Theo yêu cầu người dùng, không chạy lại train sau các bugfix cuối. Artifact/số liệu từ lần train trước không được coi là xác nhận của phiên bản code cuối.
