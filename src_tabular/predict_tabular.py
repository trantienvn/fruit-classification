"""
predict_tabular.py — Dự đoán và đánh giá chất lượng trái cây dựa trên bảng số liệu thuộc tính vật lý / hóa học.

Sử dụng mô hình học máy dạng bảng (RandomForest/SVM/GB) đã được huấn luyện tại models_tabular/.
Đầu vào gồm 7 thuộc tính:
    - Size (Kích thước)
    - Weight (Khối lượng)
    - Sweetness (Độ ngọt)
    - Crunchiness (Độ giòn)
    - Juiciness (Độ mọng nước)
    - Acidity (Độ axit)
    - Ripeness (Độ chín)
Đầu ra:
    - Quality: Good (Tốt - Đạt chuẩn) / Bad (Kém - Không đạt)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src_tabular.dataset import FEATURE_LABELS_VI, FEATURE_NAMES

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("models_tabular")


class FruitQualityClassifierTabular:
    """Lớp dự đoán chất lượng trái cây từ dữ liệu dạng bảng."""

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.pipeline = None
        self.config: dict[str, Any] = {}
        self.classes: list[str] = ["Bad", "Good"]
        self.feature_names: list[str] = FEATURE_NAMES
        self.is_ready: bool = False

    def load(self, model_dir: Path | str | None = None) -> None:
        if model_dir is not None:
            self.model_dir = Path(model_dir)

        model_path = self.model_dir / "classifier_tabular.joblib"
        config_path = self.model_dir / "config.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy mô hình bảng tại: {model_path}. "
                "Vui lòng chạy 'python -m src_tabular.train_tabular' trước."
            )

        self.pipeline = joblib.load(model_path)
        logger.info(f"[predict_tabular] Đã tải pipeline từ {model_path}")

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            self.classes = self.config.get("classes", ["Bad", "Good"])
            self.feature_names = self.config.get("features", FEATURE_NAMES)
            logger.info(f"[predict_tabular] Cấu hình mô hình: {self.config.get('model_name')}")

        self.is_ready = True

    def predict(self, features: dict[str, float] | list[float] | np.ndarray) -> dict[str, Any]:
        """Dự đoán chất lượng trái cây từ các thuộc tính đầu vào."""
        if not self.is_ready or self.pipeline is None:
            self.load()

        # Chuẩn hóa vector đầu vào
        if isinstance(features, dict):
            # Nhận dạng case-insensitive hoặc map từ tiếng Anh
            norm_dict = {k.lower(): float(v) for k, v in features.items()}
            vector = []
            for fname in self.feature_names:
                fl = fname.lower()
                if fl in norm_dict:
                    vector.append(norm_dict[fl])
                else:
                    raise ValueError(f"Thiếu thuộc tính bắt buộc: '{fname}'")
            feature_vector = np.array(vector, dtype=np.float32).reshape(1, -1)
            raw_input_dict = {fname: float(vector[i]) for i, fname in enumerate(self.feature_names)}
        elif isinstance(features, (list, tuple, np.ndarray)):
            arr = np.asarray(features, dtype=np.float32).flatten()
            if len(arr) != len(self.feature_names):
                raise ValueError(
                    f"Cần chính xác {len(self.feature_names)} thuộc tính ({self.feature_names}), nhận được {len(arr)}"
                )
            feature_vector = arr.reshape(1, -1)
            raw_input_dict = {fname: float(arr[i]) for i, fname in enumerate(self.feature_names)}
        else:
            raise TypeError("features phải là dict, list hoặc numpy array.")

        # Dự đoán
        y_raw = self.pipeline.predict(feature_vector)[0]
        is_good = (y_raw == 1 or str(y_raw).strip().lower() == "good")
        predicted_class = "Good" if is_good else "Bad"
        
        # Xác suất nếu mô hình hỗ trợ
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(feature_vector)[0]
            # classes_ trong sklearn thường là [0, 1]
            if len(probs) >= 2:
                prob_bad = float(probs[0])
                prob_good = float(probs[1])
            else:
                prob_good = float(probs[0]) if is_good else 0.0
                prob_bad = 1.0 - prob_good

            prob_dict = {"Bad": prob_bad, "Good": prob_good}
            confidence = prob_good if is_good else prob_bad
        else:
            prob_dict = {"Bad": 0.0 if is_good else 1.0, "Good": 1.0 if is_good else 0.0}
            confidence = 1.0

        label_vi = "Tốt (Đạt chuẩn)" if is_good else "Kém (Không đạt)"

        # Phân tích đánh giá từng thuộc tính
        assessments = self._assess_features(raw_input_dict)

        return {
            "predicted_class": predicted_class,
            "label_vi": label_vi,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "inputs": raw_input_dict,
            "inputs_vi": {
                FEATURE_LABELS_VI.get(k, k): v for k, v in raw_input_dict.items()
            },
            "assessments": assessments,
            "model_name": self.config.get("model_name", "TabularClassifier"),
        }

    def _assess_features(self, inputs: dict[str, float]) -> list[dict[str, str]]:
        """Nhận xét chi tiết từng thuộc tính để người dùng hiểu kết quả."""
        assessments = []
        for feat, val in inputs.items():
            vi_name = FEATURE_LABELS_VI.get(feat, feat)
            status = "Bình thường"
            detail = ""
            if feat == "Sweetness":
                if val > 1.0:
                    status = "Rất ngọt (Ưu điểm)"
                elif val < -1.0:
                    status = "Nhạt / Chua (Cần lưu ý)"
            elif feat == "Acidity":
                if val > 1.5:
                    status = "Axit cao (Khá chua)"
                elif val < -1.0:
                    status = "Axit thấp"
            elif feat == "Ripeness":
                if val > 1.0:
                    status = "Chín mọng"
                elif val < -1.0:
                    status = "Còn xanh"
                else:
                    status = "Độ chín vừa phải"
            elif feat == "Juiciness":
                if val > 1.0:
                    status = "Nhiều nước, mọng"
                elif val < -1.0:
                    status = "Khô / ít nước"
            elif feat == "Crunchiness":
                if val > 1.0:
                    status = "Giòn rụm"
                elif val < -1.0:
                    status = "Mềm / xốp"

            assessments.append({
                "feature": feat,
                "feature_vi": vi_name,
                "value": round(val, 2),
                "status": status,
            })
        return assessments


def main():
    parser = argparse.ArgumentParser(description="Dự đoán chất lượng trái cây từ số liệu")
    parser.add_argument("--size", type=float, default=0.5, help="Kích thước (Size)")
    parser.add_argument("--weight", type=float, default=1.0, help="Khối lượng (Weight)")
    parser.add_argument("--sweetness", type=float, default=1.5, help="Độ ngọt (Sweetness)")
    parser.add_argument("--crunchiness", type=float, default=0.8, help="Độ giòn (Crunchiness)")
    parser.add_argument("--juiciness", type=float, default=1.2, help="Độ mọng nước (Juiciness)")
    parser.add_argument("--acidity", type=float, default=-0.5, help="Độ axit (Acidity)")
    parser.add_argument("--ripeness", type=float, default=1.0, help="Độ chín (Ripeness)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    clf = FruitQualityClassifierTabular(model_dir=args.model_dir)
    clf.load()
    input_data = {
        "Size": args.size,
        "Weight": args.weight,
        "Sweetness": args.sweetness,
        "Crunchiness": args.crunchiness,
        "Juiciness": args.juiciness,
        "Acidity": args.acidity,
        "Ripeness": args.ripeness,
    }
    result = clf.predict(input_data)
    print("\n" + "=" * 50)
    print("KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG TRÁI CÂY (HỌC MÁY BẢNG SỐ LIỆU)")
    print("=" * 50)
    print(f"Chất lượng dự đoán : {result['predicted_class']} ({result['label_vi']})")
    print(f"Độ tin cậy        : {result['confidence'] * 100:.2f}%")
    print(f"Xác suất chi tiết : {result['probabilities']}")
    print("-" * 50)
    print("Đánh giá thuộc tính:")
    for a in result["assessments"]:
        print(f"  - {a['feature_vi']} ({a['feature']}): {a['value']} -> {a['status']}")


if __name__ == "__main__":
    main()
