# Vấn đề cần nhớ

- **Chưa có lỗi đang chặn.**
- GridSearch SVM dùng `CalibratedClassifierCV`; tham số cần đặt lồng theo
  `estimator__*`. Regression test đã xác nhận tuning, fit và predict hoạt động.
- Test suite phát một cảnh báo deprecation của `starlette.testclient` khi dùng
  `httpx`; không làm test thất bại.
