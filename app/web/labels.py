"""Nhãn hiển thị tiếng Việt và thông tin đánh giá chất lượng / độ chín trái cây.
Cung cấp giải thích chi tiết cho 2 thuộc tính: Độ chín (Ripeness) và Chất lượng (Quality).
"""

from __future__ import annotations
from typing import Any

FRUIT_NAMES = ["cam", "chuoi", "xoai"]
RIPENESS_NAMES = ["chin", "chua-chin", "hong"]

FRUIT_LABELS: dict[str, str] = {
    "Orange": "Cam",
    "Banana": "Chuối",
    "Mango": "Xoài",
    "Apple": "Táo",
    "cam": "Cam",
    "chuoi": "Chuối",
    "xoai": "Xoài",
    "tao": "Táo",
}

RIPENESS_LABELS: dict[str, str] = {
    "Unripe": "Chưa chín (Xanh)",
    "Ripe": "Chín",
    "Spoiled": "Hỏng / Quá chín",
    "Overripe": "Quá chín",
    "chin": "Chín",
    "chua-chin": "Chưa chín (Xanh)",
    "chua_chin": "Chưa chín (Xanh)",
    "hong": "Hỏng / Hư thối",
}

FULL_CLASS_MAP_VI: dict[str, str] = {
    "cam-chin": "Cam · Chín",
    "cam-chua-chin": "Cam · Chưa chín (Xanh)",
    "cam-hong": "Cam · Bị hỏng / Nấm mốc",
    "chuoi-chin": "Chuối · Chín",
    "chuoi-chua-chin": "Chuối · Chưa chín (Xanh)",
    "chuoi-hong": "Chuối · Bị hỏng / Quá thâm",
    "xoai-chin": "Xoài · Chín",
    "xoai-chua-chin": "Xoài · Chưa chín (Xanh)",
    "xoai-hong": "Xoài · Bị hỏng / Thối úng",
}

# Chi tiết phân tích đa thuộc tính (Loại quả, Độ chín, Chất lượng thương phẩm, Khuyến nghị)
DETAILED_ANALYSIS: dict[str, dict[str, Any]] = {
    "cam-chin": {
        "fruit_type": "cam",
        "fruit_type_vi": "Cam",
        "ripeness": "chin",
        "ripeness_vi": "Chín",
        "quality": "Đạt chuẩn (Tốt - Loại 1)",
        "quality_code": "good",
        "recommendation": "Trái cây ở độ chín hoàn hảo, vị ngọt thanh và giàu vitamin C. Sử dụng trực tiếp, ép nước trong 3-5 ngày hoặc bảo quản mát 4-8°C.",
    },
    "cam-chua-chin": {
        "fruit_type": "cam",
        "fruit_type_vi": "Cam",
        "ripeness": "chua-chin",
        "ripeness_vi": "Chưa chín (Xanh)",
        "quality": "Đạt chuẩn lưu kho (Tốt - Loại 2)",
        "quality_code": "unripe",
        "recommendation": "Vỏ quả còn sắc xanh, hàm lượng axit tự nhiên cao. Thích hợp lưu kho vận chuyển đường dài hoặc để nơi thoáng khí 3-4 ngày cho quả mọng nước hơn.",
    },
    "cam-hong": {
        "fruit_type": "cam",
        "fruit_type_vi": "Cam",
        "ripeness": "hong",
        "ripeness_vi": "Bị hỏng / Nấm mốc",
        "quality": "Không đạt chuẩn (Kém - Loại bỏ)",
        "quality_code": "bad",
        "recommendation": "Vỏ quả bị úng đốm nâu hoặc có dấu vết nấm mốc vi sinh. Cần loại bỏ ngay để tránh lây nhiễm sang các quả xung quanh, không sử dụng làm thực phẩm.",
    },
    "chuoi-chin": {
        "fruit_type": "chuoi",
        "fruit_type_vi": "Chuối",
        "ripeness": "chin",
        "ripeness_vi": "Chín",
        "quality": "Đạt chuẩn (Tốt - Loại 1)",
        "quality_code": "good",
        "recommendation": "Vỏ vàng đều, điểm đốm trứng cuốc tự nhiên, hàm lượng đường và chất chống oxy hóa cao nhất. Sử dụng ngay, làm sinh tố hoặc bánh chuối dinh dưỡng.",
    },
    "chuoi-chua-chin": {
        "fruit_type": "chuoi",
        "fruit_type_vi": "Chuối",
        "ripeness": "chua-chin",
        "ripeness_vi": "Chưa chín (Xanh)",
        "quality": "Đạt chuẩn lưu kho (Tốt - Loại 2)",
        "quality_code": "unripe",
        "recommendation": "Quả còn cứng và giàu tinh bột kháng. Nên để ở nhiệt độ phòng từ 2-3 ngày hoặc ủ kín với quả táo để kích thích khí ethylene giúp quả chín tự nhiên.",
    },
    "chuoi-hong": {
        "fruit_type": "chuoi",
        "fruit_type_vi": "Chuối",
        "ripeness": "hong",
        "ripeness_vi": "Bị hỏng / Quá thâm",
        "quality": "Không đạt chuẩn (Kém - Loại bỏ)",
        "quality_code": "bad",
        "recommendation": "Mô quả đã bị nhũn nát, vỏ thâm đen hoàn toàn và có hiện tượng lên men chua. Không đạt tiêu chuẩn an toàn vệ sinh thực phẩm.",
    },
    "xoai-chin": {
        "fruit_type": "xoai",
        "fruit_type_vi": "Xoài",
        "ripeness": "chin",
        "ripeness_vi": "Chín",
        "quality": "Đạt chuẩn (Tốt - Loại 1)",
        "quality_code": "good",
        "recommendation": "Thịt quả mềm ngọt, tỏa hương thơm đặc trưng, màu sắc tươi sáng. Thích hợp dùng tráng miệng trực tiếp hoặc chế biến các món ngọt.",
    },
    "xoai-chua-chin": {
        "fruit_type": "xoai",
        "fruit_type_vi": "Xoài",
        "ripeness": "chua-chin",
        "ripeness_vi": "Chưa chín (Xanh)",
        "quality": "Đạt chuẩn tiêu thụ (Tốt - Loại 2)",
        "quality_code": "unripe",
        "recommendation": "Vỏ quả xanh, thịt giòn chua. Có thể chế biến món nộm, gỏi, chấm gia vị hoặc để nơi ấm 2-4 ngày để quả chín ngọt.",
    },
    "xoai-hong": {
        "fruit_type": "xoai",
        "fruit_type_vi": "Xoài",
        "ripeness": "hong",
        "ripeness_vi": "Bị hỏng / Thối úng",
        "quality": "Không đạt chuẩn (Kém - Loại bỏ)",
        "quality_code": "bad",
        "recommendation": "Vỏ loang thâm đen, thịt quả úng nước và có mùi khét/lên men vi khuẩn. Cần tiêu hủy hoặc dùng làm phân bón hữu cơ, không được ăn.",
    },
}


def localized_class_name(class_name: str) -> str:
    """Dịch tên lớp đầy đủ sang tiếng Việt (hỗ trợ cả kebab-case và space-separated)."""
    clean_name = str(class_name).strip()
    if clean_name in FULL_CLASS_MAP_VI:
        return FULL_CLASS_MAP_VI[clean_name]

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


def get_fruit_analysis(class_name: str) -> dict[str, Any]:
    """Trả về phân tích toàn diện 2 thuộc tính: Độ chín và Chất lượng cùng lời khuyên."""
    clean_name = str(class_name).strip().lower()
    if clean_name in DETAILED_ANALYSIS:
        item = dict(DETAILED_ANALYSIS[clean_name])
        item["label_vi"] = FULL_CLASS_MAP_VI.get(clean_name, clean_name)
        return item

    # Phân tích dự phòng nếu tên lớp lạ
    parts = clean_name.split("-", 1) if "-" in clean_name else clean_name.split(" ", 1)
    fruit = parts[0] if parts else "trai-cay"
    ripeness = parts[1] if len(parts) > 1 else "chin"

    fruit_vi = FRUIT_LABELS.get(fruit, fruit.title())
    ripeness_vi = RIPENESS_LABELS.get(ripeness, ripeness.title())
    is_spoiled = "hong" in ripeness or "spoil" in ripeness

    return {
        "fruit_type": fruit,
        "fruit_type_vi": fruit_vi,
        "ripeness": ripeness,
        "ripeness_vi": ripeness_vi,
        "quality": "Không đạt chuẩn (Kém)" if is_spoiled else "Đạt chuẩn chất lượng",
        "quality_code": "bad" if is_spoiled else "good",
        "recommendation": "Kiểm tra kỹ trước khi sử dụng thực phẩm.",
        "label_vi": f"{fruit_vi} · {ripeness_vi}",
    }
