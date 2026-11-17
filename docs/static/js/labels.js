/**
 * labels.js — Từ điển tiếng Việt, nhãn phân tích chất lượng & độ chín cho client-side inference.
 * Port đầy đủ từ app/web/labels.py để chạy 100% trong trình duyệt.
 */

export const CLASS_NAMES = [
  "cam-chin",
  "cam-chua-chin",
  "cam-hong",
  "chuoi-chin",
  "chuoi-chua-chin",
  "chuoi-hong",
  "xoai-chin",
  "xoai-chua-chin",
  "xoai-hong",
];

export const FRUIT_NAMES = ["cam", "chuoi", "xoai"];
export const RIPENESS_NAMES = ["chin", "chua-chin", "hong"];

export const FRUIT_LABELS = {
  Orange: "Cam",
  Banana: "Chuối",
  Mango: "Xoài",
  Apple: "Táo",
  cam: "Cam",
  chuoi: "Chuối",
  xoai: "Xoài",
  tao: "Táo",
};

export const RIPENESS_LABELS = {
  Unripe: "Chưa chín (Xanh)",
  Ripe: "Chín",
  Spoiled: "Hỏng / Quá chín",
  Overripe: "Quá chín",
  chin: "Chín",
  "chua-chin": "Chưa chín (Xanh)",
  "chua_chin": "Chưa chín (Xanh)",
  hong: "Hỏng / Hư thối",
};

export const FULL_CLASS_MAP_VI = {
  "cam-chin": "Cam · Chín",
  "cam-chua-chin": "Cam · Chưa chín (Xanh)",
  "cam-hong": "Cam · Bị hỏng / Nấm mốc",
  "chuoi-chin": "Chuối · Chín",
  "chuoi-chua-chin": "Chuối · Chưa chín (Xanh)",
  "chuoi-hong": "Chuối · Bị hỏng / Quá thâm",
  "xoai-chin": "Xoài · Chín",
  "xoai-chua-chin": "Xoài · Chưa chín (Xanh)",
  "xoai-hong": "Xoài · Bị hỏng / Thối úng",
};

export const DETAILED_ANALYSIS = {
  "cam-chin": {
    fruit_type: "cam",
    fruit_type_vi: "Cam",
    ripeness: "chin",
    ripeness_vi: "Chín",
    quality: "Đạt chuẩn (Tốt - Loại 1)",
    quality_code: "good",
    recommendation: "Trái cây ở độ chín hoàn hảo, vị ngọt thanh và giàu vitamin C. Sử dụng trực tiếp, ép nước trong 3-5 ngày hoặc bảo quản mát 4-8°C.",
  },
  "cam-chua-chin": {
    fruit_type: "cam",
    fruit_type_vi: "Cam",
    ripeness: "chua-chin",
    ripeness_vi: "Chưa chín (Xanh)",
    quality: "Đạt chuẩn lưu kho (Tốt - Loại 2)",
    quality_code: "unripe",
    recommendation: "Vỏ quả còn sắc xanh, hàm lượng axit tự nhiên cao. Thích hợp lưu kho vận chuyển đường dài hoặc để nơi thoáng khí 3-4 ngày cho quả mọng nước hơn.",
  },
  "cam-hong": {
    fruit_type: "cam",
    fruit_type_vi: "Cam",
    ripeness: "hong",
    ripeness_vi: "Bị hỏng / Nấm mốc",
    quality: "Không đạt chuẩn (Kém - Loại bỏ)",
    quality_code: "bad",
    recommendation: "Vỏ quả bị úng đốm nâu hoặc có dấu vết nấm mốc vi sinh. Cần loại bỏ ngay để tránh lây nhiễm sang các quả xung quanh, không sử dụng làm thực phẩm.",
  },
  "chuoi-chin": {
    fruit_type: "chuoi",
    fruit_type_vi: "Chuối",
    ripeness: "chin",
    ripeness_vi: "Chín",
    quality: "Đạt chuẩn (Tốt - Loại 1)",
    quality_code: "good",
    recommendation: "Vỏ vàng đều, điểm đốm trứng cuốc tự nhiên, hàm lượng đường và chất chống oxy hóa cao nhất. Sử dụng ngay, làm sinh tố hoặc bánh chuối dinh dưỡng.",
  },
  "chuoi-chua-chin": {
    fruit_type: "chuoi",
    fruit_type_vi: "Chuối",
    ripeness: "chua-chin",
    ripeness_vi: "Chưa chín (Xanh)",
    quality: "Đạt chuẩn lưu kho (Tốt - Loại 2)",
    quality_code: "unripe",
    recommendation: "Quả còn cứng và giàu tinh bột kháng. Nên để ở nhiệt độ phòng từ 2-3 ngày hoặc ủ kín với quả táo để kích thích khí ethylene giúp quả chín tự nhiên.",
  },
  "chuoi-hong": {
    fruit_type: "chuoi",
    fruit_type_vi: "Chuối",
    ripeness: "hong",
    ripeness_vi: "Bị hỏng / Quá thâm",
    quality: "Không đạt chuẩn (Kém - Loại bỏ)",
    quality_code: "bad",
    recommendation: "Mô quả đã bị nhũn nát, vỏ thâm đen hoàn toàn và có hiện tượng lên men chua. Không đạt tiêu chuẩn an toàn vệ sinh thực phẩm.",
  },
  "xoai-chin": {
    fruit_type: "xoai",
    fruit_type_vi: "Xoài",
    ripeness: "chin",
    ripeness_vi: "Chín",
    quality: "Đạt chuẩn (Tốt - Loại 1)",
    quality_code: "good",
    recommendation: "Thịt quả mềm ngọt, tỏa hương thơm đặc trưng, màu sắc tươi sáng. Thích hợp dùng tráng miệng trực tiếp hoặc chế biến các món ngọt.",
  },
  "xoai-chua-chin": {
    fruit_type: "xoai",
    fruit_type_vi: "Xoài",
    ripeness: "chua-chin",
    ripeness_vi: "Chưa chín (Xanh)",
    quality: "Đạt chuẩn tiêu thụ (Tốt - Loại 2)",
    quality_code: "unripe",
    recommendation: "Vỏ quả xanh, thịt giòn chua. Có thể chế biến món nộm, gỏi, chấm gia vị hoặc để nơi ấm 2-4 ngày để quả chín ngọt.",
  },
  "xoai-hong": {
    fruit_type: "xoai",
    fruit_type_vi: "Xoài",
    ripeness: "hong",
    ripeness_vi: "Bị hỏng / Thối úng",
    quality: "Không đạt chuẩn (Kém - Loại bỏ)",
    quality_code: "bad",
    recommendation: "Vỏ loang thâm đen, thịt quả úng nước và có mùi khét/lên men vi khuẩn. Cần tiêu hủy hoặc dùng làm phân bón hữu cơ, không được ăn.",
  },
};

export function localizedClassName(className) {
  const clean = String(className || "").trim().toLowerCase();
  if (FULL_CLASS_MAP_VI[clean]) return FULL_CLASS_MAP_VI[clean];

  if (clean.includes("-")) {
    const parts = clean.split("-");
    const fruit = FRUIT_LABELS[parts[0]] || parts[0];
    const ripeness = RIPENESS_LABELS[parts.slice(1).join("-")] || parts.slice(1).join("-");
    return `${fruit} · ${ripeness}`;
  }
  return clean;
}

export function getFruitAnalysis(className) {
  const clean = String(className || "").trim().toLowerCase();
  if (DETAILED_ANALYSIS[clean]) {
    return {
      ...DETAILED_ANALYSIS[clean],
      label_vi: FULL_CLASS_MAP_VI[clean] || clean,
    };
  }

  // Fallback
  const parts = clean.split("-");
  const fruit = parts[0] || "trai-cay";
  const ripeness = parts[1] || "chin";
  const isSpoiled = ripeness.includes("hong") || ripeness.includes("spoil");

  return {
    fruit_type: fruit,
    fruit_type_vi: FRUIT_LABELS[fruit] || fruit,
    ripeness: ripeness,
    ripeness_vi: RIPENESS_LABELS[ripeness] || ripeness,
    quality: isSpoiled ? "Không đạt chuẩn (Kém - Loại bỏ)" : "Đạt chuẩn chất lượng",
    quality_code: isSpoiled ? "bad" : "good",
    recommendation: "Kiểm tra kỹ tình trạng sản phẩm trước khi sử dụng.",
    label_vi: `${FRUIT_LABELS[fruit] || fruit} · ${RIPENESS_LABELS[ripeness] || ripeness}`,
  };
}

export function formatPredictionData(arg1, arg2, arg3, arg4) {
  let className = "";
  let probabilitiesObj = {};
  let latencyMs = 0;
  let filename = "anh-trai-cay.jpg";

  if (
    Array.isArray(arg1) ||
    (arg1 && typeof arg1 === "object" && typeof arg1.length === "number" && !(arg1 instanceof Element))
  ) {
    // Trường hợp: formatPredictionData(probsArray, classNames, latencyMs, filename)
    const probsArray = Array.from(arg1);
    const names = Array.isArray(arg2) && arg2.length ? arg2 : CLASS_NAMES;
    latencyMs = typeof arg3 === "number" ? arg3 : 0;
    filename = typeof arg4 === "string" && arg4.trim() ? arg4 : "anh-trai-cay.jpg";

    let maxIdx = 0;
    let maxVal = -Infinity;
    for (let i = 0; i < names.length; i++) {
      const p = Number(probsArray[i] ?? 0);
      probabilitiesObj[names[i]] = p;
      if (p > maxVal) {
        maxVal = p;
        maxIdx = i;
      }
    }
    className = names[maxIdx] || CLASS_NAMES[0];
  } else if (arg1 && typeof arg1 === "object") {
    // Trường hợp: formatPredictionData({ className, probabilities, latencyMs, filename })
    latencyMs =
      typeof arg1.latencyMs === "number"
        ? arg1.latencyMs
        : typeof arg1.processing_time_ms === "number"
        ? arg1.processing_time_ms
        : 0;
    filename = arg1.filename || "anh-trai-cay.jpg";

    const rawProbs = arg1.probabilities;
    if (Array.isArray(rawProbs) || (rawProbs && typeof rawProbs.length === "number")) {
      const probsArray = Array.from(rawProbs);
      const names = Array.isArray(arg2) && arg2.length ? arg2 : CLASS_NAMES;
      let maxIdx = 0;
      let maxVal = -Infinity;
      for (let i = 0; i < names.length; i++) {
        const p = Number(probsArray[i] ?? 0);
        probabilitiesObj[names[i]] = p;
        if (p > maxVal) {
          maxVal = p;
          maxIdx = i;
        }
      }
      className = arg1.className || names[maxIdx] || CLASS_NAMES[0];
    } else if (rawProbs && typeof rawProbs === "object") {
      probabilitiesObj = { ...rawProbs };
      if (arg1.className) {
        className = arg1.className;
      } else {
        let maxVal = -Infinity;
        for (const [k, v] of Object.entries(probabilitiesObj)) {
          const numV = Number(v);
          if (numV > maxVal) {
            maxVal = numV;
            className = k;
          }
        }
      }
    } else {
      className = arg1.className || CLASS_NAMES[0];
      probabilitiesObj = { [className]: 1.0 };
    }
  } else {
    className = CLASS_NAMES[0];
    probabilitiesObj = { [className]: 1.0 };
  }

  if (!className) className = CLASS_NAMES[0];

  const analysis = getFruitAnalysis(className);
  const classId = CLASS_NAMES.indexOf(className);
  const confidence =
    probabilitiesObj[className] != null
      ? Number(probabilitiesObj[className])
      : Object.values(probabilitiesObj).length > 0
      ? Math.max(...Object.values(probabilitiesObj).map(Number))
      : 1.0;

  const top3 = Object.entries(probabilitiesObj)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([lbl, p]) => ({
      label: lbl,
      label_vi: FULL_CLASS_MAP_VI[lbl] || localizedClassName(lbl),
      p: Number(p),
    }));

  return {
    id: Date.now(),
    filename,
    predicted_class: classId >= 0 ? classId : 0,
    class_name: className,
    class_name_vi: analysis.label_vi || localizedClassName(className),
    fruit_type: analysis.fruit_type,
    fruit_type_vi: analysis.fruit_type_vi,
    ripeness: analysis.ripeness,
    ripeness_vi: analysis.ripeness_vi,
    quality: analysis.quality,
    quality_code: analysis.quality_code,
    recommendation: analysis.recommendation,
    confidence: Number(confidence),
    probabilities: probabilitiesObj,
    top3,
    processing_time_ms: Number(latencyMs),
    created_at: new Date().toISOString(),
  };
}
