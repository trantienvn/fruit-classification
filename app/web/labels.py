"""Nhãn hiển thị tiếng Việt dùng chung cho response của Web API."""

from __future__ import annotations

FRUIT_LABELS = {
    "Orange": "Cam",
    "Banana": "Chuối",
    "Mango": "Xoài",
    "Apple": "Táo",
}
RIPENESS_LABELS = {
    "Unripe": "Chưa chín",
    "Ripe": "Chín",
    "Spoiled": "Hỏng",
    "Overripe": "Quá chín",
}


def localized_class_name(class_name: str) -> str:
    """Dịch tên lớp đầy đủ, ví dụ ``Orange Spoiled`` thành ``Cam · Hỏng``."""
    parts = class_name.split(maxsplit=1)
    if len(parts) != 2:
        return class_name

    fruit, ripeness = parts
    fruit_label = FRUIT_LABELS.get(fruit, fruit)
    ripeness_label = RIPENESS_LABELS.get(ripeness, ripeness)
    return f"{fruit_label} · {ripeness_label}"
