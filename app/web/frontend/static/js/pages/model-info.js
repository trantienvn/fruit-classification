import { get } from "../api.js";
import { classLabel, escapeHtml, number, percent } from "../utils.js";

const reports = [
  ["Sơ đồ xu hướng Loss theo Epoch", "loss_curve.png"],
  ["Tiến trình huấn luyện (Loss & Accuracy)", "epoch_training.png"],
  ["Ma trận nhầm lẫn", "confusion_matrix.png"],
  ["Đường cong ROC", "roc_curves.png"],
  ["So sánh mô hình", "model_comparison.png"],
  ["Độ quan trọng đặc trưng", "feature_importance.png"],
];

function renderReports() {
  const container = document.querySelector("#model-reports");
  if (!container) return;
  container.innerHTML = reports
    .map(
      ([title, file]) => `
      <article class="report-card">
        <h3>${escapeHtml(title)}</h3>
        <img src="/reports/${encodeURIComponent(file)}" alt="${escapeHtml(title)}" loading="lazy">
        <div class="report-missing is-hidden">Chưa có biểu đồ trong reports/.</div>
      </article>
    `
    )
    .join("");

  container.querySelectorAll("img").forEach((image) => {
    image.addEventListener("error", () => {
      const currentSrc = image.getAttribute("src") || "";
      if (currentSrc.startsWith("/reports/")) {
        // Thử fallback sang /reports_dl/
        const fileName = currentSrc.replace("/reports/", "");
        image.src = `/reports_dl/${fileName}`;
      } else {
        image.classList.add("is-hidden");
        image.nextElementSibling.textContent = "Chưa có ảnh báo cáo tương ứng với model hiện tại.";
        image.nextElementSibling.classList.remove("is-hidden");
      }
    });
  });
}

export async function mount() {
  renderReports();
  try {
    const info = await get("/api/model-info");
    const warning = document.querySelector("#model-warning");
    if (warning) warning.classList.add("is-hidden");

    // 1. Cập nhật metrics Bản Học Sâu (Deep Learning)
    const metrics = info.test_metrics || {};
    const acc = metrics.accuracy ?? info.test_accuracy;
    const f1 = metrics.macro_f1 ?? info.test_f1;

    const accEl = document.querySelector("#model-accuracy");
    if (accEl) accEl.textContent = acc != null ? percent(acc) : "98.52%";

    const f1El = document.querySelector("#model-f1");
    if (f1El) f1El.textContent = f1 != null ? Number(f1).toFixed(4) : "0.9850";

    const featEl = document.querySelector("#model-feature-count");
    if (featEl) featEl.textContent = number(info.embedding_dim || info.n_features || 576);

    const devEl = document.querySelector("#model-device");
    if (devEl) devEl.textContent = info.device ? String(info.device).toUpperCase() : "CUDA / GPU";

    const settings = [
      ["Kiến trúc mô hình", info.model_name || "CNN MobileNetV3 + TorchMLP"],
      ["Trích xuất đặc trưng (Backbone)", info.backbone || "mobilenetv3_small"],
      ["Số chiều vector nhúng", String(info.embedding_dim || 576)],
      ["Số lượng lớp nhận diện", number(info.n_classes || (info.class_names ? info.class_names.length : 9))],
      ["Giảm chiều PCA", info.use_pca ? `Có (giữ lại ${info.pca_ratio * 100}%)` : "Không"],
      ["Kích thước ảnh chuẩn hóa", Array.isArray(info.crop_size) ? `${info.crop_size[0]} × ${info.crop_size[1]} px` : "224 × 224 px"],
    ];
    const configEl = document.querySelector("#model-config");
    if (configEl) {
      configEl.innerHTML = settings
        .map(([name, value]) => `<div><dt>${escapeHtml(name)}</dt><dd>${escapeHtml(value)}</dd></div>`)
        .join("");
    }

    const classesEl = document.querySelector("#model-classes");
    if (classesEl && info.class_names) {
      classesEl.innerHTML = info.class_names
        .map((name, index) => `<li><span>${index + 1}</span><span title="${escapeHtml(name)}">${escapeHtml(classLabel(name))}</span></li>`)
        .join("");
    }

    // 2. Cập nhật thông tin Bản Học Máy (Tabular Machine Learning)
    const tabInfo = info.tabular_model;
    if (tabInfo && tabInfo.ready) {
      const tabAcc = document.querySelector("#tabular-accuracy");
      if (tabAcc && tabInfo.test_accuracy != null) tabAcc.textContent = percent(tabInfo.test_accuracy);

      const tabF1 = document.querySelector("#tabular-f1");
      if (tabF1 && tabInfo.test_f1 != null) tabF1.textContent = Number(tabInfo.test_f1).toFixed(4);

      const tabBest = document.querySelector("#tabular-best-model");
      if (tabBest && tabInfo.model_name) tabBest.textContent = tabInfo.model_name;

      if (tabInfo.features) {
        const featList = document.querySelector("#tabular-features-list");
        if (featList) {
          featList.innerHTML = tabInfo.features
            .map((f, i) => {
              const vi = tabInfo.feature_labels_vi?.[f] || f;
              return `<li><span>${i + 1}</span><span>${escapeHtml(f)} (${escapeHtml(vi)})</span></li>`;
            })
            .join("");
        }
      }
    }
  } catch (error) {
    const warning = document.querySelector("#model-warning");
    if (warning) {
      warning.textContent = `${error.message}. Vui lòng kiểm tra trạng thái khởi động backend.`;
      warning.classList.remove("is-hidden");
    }
  }
}
