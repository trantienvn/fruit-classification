import { get } from "../api.js";
import { classLabel, escapeHtml, number, percent } from "../utils.js";

const reports = [
  ["Ma trận nhầm lẫn", "confusion_matrix.png"],
  ["Đường cong ROC", "roc_curves.png"],
  ["So sánh mô hình", "model_comparison.png"],
  ["Độ quan trọng đặc trưng", "feature_importance.png"],
];

function renderReports() {
  const container = document.querySelector("#model-reports");
  container.innerHTML = reports.map(([title, file]) => `<article class="report-card"><h3>${escapeHtml(title)}</h3><img src="/reports/${encodeURIComponent(file)}" alt="${escapeHtml(title)}" loading="lazy"><div class="report-missing is-hidden">Chưa có biểu đồ trong reports/.</div></article>`).join("");
  container.querySelectorAll("img").forEach((image) => image.addEventListener("error", () => {
    image.classList.add("is-hidden");
    image.nextElementSibling.classList.remove("is-hidden");
  }, { once: true }));
}

export async function mount() {
  renderReports();
  try {
    const info = await get("/api/model-info");
    const metrics = info.test_metrics || {};
    document.querySelector("#model-warning").classList.add("is-hidden");
    document.querySelector("#model-accuracy").textContent = percent(metrics.accuracy);
    document.querySelector("#model-f1").textContent = Number.isFinite(metrics.macro_f1) ? Number(metrics.macro_f1).toFixed(4) : "—";
    document.querySelector("#model-train-count").textContent = number(info.n_train);
    document.querySelector("#model-feature-count").textContent = number(info.n_features);

    const settings = [
      ["Thuật toán", info.model_name || "—"],
      ["Số lớp", number(info.n_classes)],
      ["PCA", info.use_pca ? "Có" : "Không"],
      ["Kích thước crop", Array.isArray(info.crop_size) ? `${info.crop_size[0]} × ${info.crop_size[1]} px` : "—"],
    ];
    document.querySelector("#model-config").innerHTML = settings.map(([name, value]) => `<div><dt>${escapeHtml(name)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
    document.querySelector("#model-classes").innerHTML = (info.class_names || []).map((name, index) => `<li><span>${index + 1}</span><span title="${escapeHtml(name)}">${escapeHtml(classLabel(name))}</span></li>`).join("");
  } catch (error) {
    document.querySelector("#model-warning").textContent = `${error.message} Thông tin metrics không được tạo khi thiếu model.`;
    document.querySelector("#model-warning").classList.remove("is-hidden");
    document.querySelector("#model-metrics").classList.add("is-hidden");
  }
}
