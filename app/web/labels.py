"""Nhãn hiển thị tiếng Việt dùng chung cho response của Web API."""

from __future__ import annotations

FRUIT_LABELS = {
    "Orange": "Cam",
    "Banana": "Chuối",
    "Mango": "Xoài",
    "Apple": "Táo",
    "cam": "Cam",
    "chuoi": "Chuối",
    "xoai": "Xoài",
    "tao": "Táo",
}

RIPENESS_LABELS = {
    "Unripe": "Chưa chín",
    "Ripe": "Chín",
    "Spoiled": "Hỏng",
    "Overripe": "Quá chín",
    "chin": "Chín",
    "chua-chin": "Chưa chín",
    "chua_chin": "Chưa chín",
    "hong": "Hỏng",
}

FULL_CLASS_MAP_VI = {
    "cam-chin": "Cam · Chín",
    "cam-chua-chin": "Cam · Chưa chín",
    "cam-hong": "Cam · Hỏng",
    "chuoi-chin": "Chuối · Chín",
    "chuoi-chua-chin": "Chuối · Chưa chín",
    "chuoi-hong": "Chuối · Hỏng",
    "xoai-chin": "Xoài · Chín",
    "xoai-chua-chin": "Xoài · Chưa chín",
    "xoai-hong": "Xoài · Hỏng",
}


def localized_class_name(class_name: str) -> str:
    """Dịch tên lớp đầy đủ sang tiếng Việt (hỗ trợ cả kebab-case và space-separated)."""
    clean_name = str(class_name).strip()
    if clean_name in FULL_CLASS_MAP_VI:
        return FULL_CLASS_MAP_VI[clean_name]

    # Xử lý kebab-case dạng <fruit>-<ripeness>
    if "-" in clean_name and " " not in clean_name:
        parts = clean_name.split("-", 1)
        fruit = FRUIT_LABELS.get(parts[0].lower(), parts[0])
        ripeness = RIPENESS_LABELS.get(parts[1].lower(), parts[1])
        return f"{fruit} · {ripeness}"

    parts = clean_name.split(maxsplit=1)
    if len(parts) == 2:
        fruit, ripeness = parts
        fruit_label = FRUIT_LABELS.get(fruit, fruit)
        ripeness_label = RIPENESS_LABELS.get(ripeness, ripeness)
        return f"{fruit_label} · {ripeness_label}"

    return clean_name
