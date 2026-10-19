"""Điểm mở rộng cho đánh giá bằng scikit-learn; chưa có checkpoint/dataset."""


def evaluate_predictions(y_true, y_pred) -> dict:
    """Tính các chỉ số phân loại cơ bản."""
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }

