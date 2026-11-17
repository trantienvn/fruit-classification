export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

const FRUIT_NAMES_MAP = {
  Orange: "Cam",
  Banana: "Chuối",
  Mango: "Xoài",
  Apple: "Táo",
  cam: "Cam",
  chuoi: "Chuối",
  xoai: "Xoài",
  tao: "Táo",
};

const RIPENESS_NAMES_MAP = {
  Unripe: "Chưa chín",
  Ripe: "Chín",
  Spoiled: "Hỏng",
  Overripe: "Quá chín",
  chin: "Chín",
  "chua-chin": "Chưa chín",
  "chua_chin": "Chưa chín",
  hong: "Hỏng",
};

const FULL_LABELS_MAP = {
  "cam-chin": "Cam · Chín",
  "cam-chua-chin": "Cam · Chưa chín (Xanh)",
  "cam-hong": "Cam · Bị hỏng",
  "chuoi-chin": "Chuối · Chín",
  "chuoi-chua-chin": "Chuối · Chưa chín (Xanh)",
  "chuoi-hong": "Chuối · Bị hỏng",
  "xoai-chin": "Xoài · Chín",
  "xoai-chua-chin": "Xoài · Chưa chín (Xanh)",
  "xoai-hong": "Xoài · Bị hỏng",
};

export function fruitLabel(value) {
  const key = String(value || "").toLowerCase();
  return FRUIT_NAMES_MAP[value] || FRUIT_NAMES_MAP[key] || value || "—";
}

export function ripenessLabel(value) {
  const key = String(value || "").toLowerCase();
  return RIPENESS_NAMES_MAP[value] || RIPENESS_NAMES_MAP[key] || value || "—";
}

export function classLabel(value) {
  const key = String(value || "").trim().toLowerCase();
  if (FULL_LABELS_MAP[key]) return FULL_LABELS_MAP[key];

  if (key.includes("-") && !key.includes(" ")) {
    const [fruit, ...ripeness] = key.split("-");
    return `${fruitLabel(fruit)} · ${ripenessLabel(ripeness.join("-"))}`;
  }

  const [fruit, ...ripeness] = String(value || "").split(" ");
  return `${fruitLabel(fruit)} · ${ripenessLabel(ripeness.join(" "))}`;
}

export function percent(value, digits = 1) {
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : "—";
}

export function number(value) {
  return Number(value || 0).toLocaleString("vi-VN");
}

export function formatDate(value, withTime = true) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("vi-VN", withTime
    ? { dateStyle: "short", timeStyle: "short" }
    : { day: "2-digit", month: "2-digit" }).format(date);
}

export function renderBars(rows, labelKey, container, options = {}) {
  if (!rows?.length) {
    container.innerHTML = '<p class="chart-empty">Chưa có dữ liệu phân loại.</p>';
    return;
  }
  const maximum = Math.max(1, ...rows.map((row) => Number(row.count) || 0));
  container.innerHTML = rows.map((row) => {
    const value = Number(row.count) || 0;
    const label = options.displayLabel ? options.displayLabel(row[labelKey]) : row[labelKey];
    const width = value ? Math.max(3, (value / maximum) * 100) : 0;
    return `<div class="chart-row">
      <span class="chart-label" title="${escapeHtml(label)}">${escapeHtml(label)}</span>
      <span class="chart-track"><span class="chart-fill" style="width:${width}%"></span></span>
      <span class="chart-value">${number(value)}</span>
    </div>`;
  }).join("");
}
