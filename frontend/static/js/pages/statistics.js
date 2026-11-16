import { get } from "../api.js";
import { classLabel, escapeHtml, fruitLabel, number, ripenessLabel } from "../utils.js";

function renderDays(rows) {
  const container = document.querySelector("#stats-day-chart");
  if (!rows?.length || rows.every((row) => !row.count)) {
    container.innerHTML = '<p class="chart-empty">Chưa có lượt phân loại trong 7 ngày gần nhất.</p>';
    return;
  }
  const max = Math.max(1, ...rows.map((row) => row.count));
  container.innerHTML = rows.map((row) => {
    const height = row.count ? Math.max(7, row.count / max * 92) : 2;
    const label = new Intl.DateTimeFormat("vi-VN", { weekday: "short" }).format(new Date(`${row.date}T12:00:00`));
    return `<div class="day-column" title="${escapeHtml(row.date)}: ${number(row.count)} lượt"><strong>${number(row.count)}</strong><div class="day-bar-wrap"><span class="day-bar" style="height:${height}px"></span></div><span>${escapeHtml(label)}</span></div>`;
  }).join("");
}

export async function mount() {
  try {
    const data = await get("/api/statistics");
    document.querySelector("#stats-total").textContent = number(data.total_predictions);
    document.querySelector("#stats-classes").textContent = `${data.unique_classes} / 9`;
    document.querySelector("#stats-latest").textContent = data.latest_at
      ? new Intl.DateTimeFormat("vi-VN", { dateStyle: "short", timeStyle: "short", timeZone: "UTC" }).format(new Date(data.latest_at))
      : "Chưa có";
    document.querySelector("#stats-class-chart").innerHTML = "";
    renderBars(data.by_class.map((row) => ({ ...row, label: classLabel(row.class_name) })), "label", document.querySelector("#stats-class-chart"));
    renderDays(data.by_day);
    renderBars(data.by_fruit.map((row) => ({ ...row, label: fruitLabel(row.fruit) })), "label", document.querySelector("#stats-fruit-chart"));
    renderBars(data.by_ripeness.map((row) => ({ ...row, label: ripenessLabel(row.ripeness) })), "label", document.querySelector("#stats-ripeness-chart"));
  } catch (error) {
    document.querySelectorAll(".chart-panel").forEach((panel) => {
      const chart = panel.querySelector(".chart-list, .day-chart");
      if (chart) chart.innerHTML = `<p class="chart-empty">${escapeHtml(error.message)}</p>`;
    });
  }
}

function renderBars(rows, labelKey, container) {
  const maximum = Math.max(1, ...rows.map((row) => Number(row.count) || 0));
  container.innerHTML = rows.map((row) => {
    const value = Number(row.count) || 0;
    const width = value ? Math.max(3, (value / maximum) * 100) : 0;
    return `<div class="chart-row"><span class="chart-label" title="${escapeHtml(row[labelKey])}">${escapeHtml(row[labelKey])}</span><span class="chart-track"><span class="chart-fill" style="width:${width}%"></span></span><span class="chart-value">${number(value)}</span></div>`;
  }).join("");
}
