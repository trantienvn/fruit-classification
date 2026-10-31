export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

export function fruitLabel(value) {
  return ({ Apple: "Táo", Banana: "Chuối", Mango: "Xoài" })[value] || value || "—";
}

export function ripenessLabel(value) {
  return ({ Unripe: "Chưa chín", Ripe: "Chín", Overripe: "Quá chín" })[value] || value || "—";
}

export function classLabel(value) {
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
