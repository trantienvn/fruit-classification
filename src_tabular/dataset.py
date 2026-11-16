"""
dataset.py — Quản lý bộ dữ liệu bảng đặc tính trái cây (Fruit Quality Dataset).

Các thuộc tính đầu vào (Inputs):
    - Size        : Kích thước
    - Weight      : Khối lượng
    - Sweetness   : Độ ngọt
    - Crunchiness : Độ giòn
    - Juiciness   : Độ mọng nước
    - Acidity     : Độ axit
    - Ripeness    : Độ chín

Thuộc tính mục tiêu (Output):
    - Quality     : Chất lượng (Good: Đạt chuẩn / Bad: Kém chất lượng)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "Size",
    "Weight",
    "Sweetness",
    "Crunchiness",
    "Juiciness",
    "Acidity",
    "Ripeness",
]

FEATURE_LABELS_VI = {
    "Size": "Kích thước",
    "Weight": "Khối lượng",
    "Sweetness": "Độ ngọt",
    "Crunchiness": "Độ giòn",
    "Juiciness": "Độ mọng nước",
    "Acidity": "Độ axit",
    "Ripeness": "Độ chín",
}

DEFAULT_TABULAR_DIR = Path("data/tabular")
DEFAULT_CSV_PATH = DEFAULT_TABULAR_DIR / "fruit_quality.csv"


def generate_fruit_quality_dataset(
    n_samples: int = 4000,
    random_state: int = 42,
    save_path: Path | None = None,
) -> Any:
    """
    Tạo bộ dữ liệu chuẩn hóa Fruit Quality với 4.000 mẫu dựa trên phân phối thực tế.

    Hàm tiềm năng:
        Score = 0.35*Size + 0.30*Weight + 0.40*Sweetness + 0.25*Crunchiness + 0.30*Juiciness - 0.20*Acidity + 0.15*Ripeness + noise
        Quality = 'Good' nếu Score >= 0, ngược lại 'Bad'
    """
    import pandas as pd
    rng = np.random.default_rng(random_state)

    # Sinh các đặc trưng liên tục chuẩn hóa (mean ~ 0, std ~ 1.5 đến 2.0)
    size = rng.normal(loc=-0.5, scale=1.9, size=n_samples)
    weight = rng.normal(loc=-1.0, scale=1.6, size=n_samples)
    sweetness = rng.normal(loc=-0.4, scale=1.9, size=n_samples)
    crunchiness = rng.normal(loc=1.0, scale=1.4, size=n_samples)
    juiciness = rng.normal(loc=0.5, scale=1.9, size=n_samples)
    acidity = rng.normal(loc=0.1, scale=2.1, size=n_samples)
    ripeness = rng.normal(loc=0.5, scale=1.8, size=n_samples)

    # Điểm đánh giá chất lượng tổng hợp
    score = (
        0.38 * size
        + 0.32 * weight
        + 0.45 * sweetness
        + 0.28 * crunchiness
        + 0.35 * juiciness
        - 0.25 * acidity
        + 0.18 * ripeness
        + rng.normal(0, 0.9, size=n_samples)
    )

    quality = np.where(score >= 0, "Good", "Bad")

    df = pd.DataFrame(
        {
            "Size": np.round(size, 4),
            "Weight": np.round(weight, 4),
            "Sweetness": np.round(sweetness, 4),
            "Crunchiness": np.round(crunchiness, 4),
            "Juiciness": np.round(juiciness, 4),
            "Acidity": np.round(acidity, 4),
            "Ripeness": np.round(ripeness, 4),
            "Quality": quality,
        }
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False, encoding="utf-8")
        logger.info(f"Đã lưu dataset bảng chất lượng trái cây vào: {save_path}")

    return df


def load_or_create_tabular_data(
    csv_path: Path = DEFAULT_CSV_PATH,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Nạp dữ liệu từ file CSV, nếu chưa có thì tự động sinh và lưu.

    Trả về:
        X (n_samples, 7), y (n_samples,), feature_names
    """
    import pandas as pd
    csv_path = Path(csv_path)
    if not csv_path.exists():
        df = generate_fruit_quality_dataset(save_path=csv_path)
    else:
        df = pd.read_csv(csv_path)

    # Đảm bảo có đủ 7 cột đặc trưng
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Tệp {csv_path} thiếu các cột: {missing}")

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y_raw = df["Quality"].astype(str).str.strip()
    y = np.where(y_raw.str.lower() == "good", 1, 0)

    return X, y, FEATURE_NAMES
