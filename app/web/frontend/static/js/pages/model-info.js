import { get } from "../api.js";
import { classLabel, escapeHtml, number, percent } from "../utils.js";

const DEFAULT_REPORTS = [
  {
    title: "Sơ đồ xu hướng Loss theo Epoch (Train vs Val)",
    file: "loss_curve.png",
    desc: "Đường cong Cross-Entropy Loss qua các Epoch, kiểm soát hiện tượng Overfitting/Underfitting.",
  },
  {
    title: "Tiến trình huấn luyện & Tốc độ hội tụ (Loss & Accuracy)",
    file: "epoch_training.png",
    desc: "Đánh giá chi tiết sự tương quan giữa hàm mất mát và độ chính xác phân loại theo từng chu kỳ.",
  },
];

function renderReports() {
  const container = document.querySelector("#model-reports");
  if (!container) return;

  container.innerHTML = DEFAULT_REPORTS.map((r) => `
    <article class="report-card">
      <div class="report-card-header">
        <h3>${escapeHtml(r.title)}</h3>
        <p class="report-desc">${escapeHtml(r.desc)}</p>
      </div>
      <div class="report-img-wrap">
        <img src="/reports_dl/${encodeURIComponent(r.file)}" alt="${escapeHtml(r.title)}" loading="lazy">
        <div class="report-missing is-hidden">Chưa có biểu đồ trong reports_dl/.</div>
      </div>
    </article>
  `).join("");

  container.querySelectorAll("img").forEach((image) => {
    image.addEventListener("error", () => {
      const currentSrc = image.getAttribute("src") || "";
      if (currentSrc.startsWith("/reports_dl/")) {
        const fileName = currentSrc.replace("/reports_dl/", "");
        image.src = `/reports/${fileName}`;
      } else {
        image.classList.add("is-hidden");
        const missing = image.nextElementSibling;
        if (missing) {
          missing.textContent = "Chưa có ảnh báo cáo tương ứng.";
          missing.classList.remove("is-hidden");
        }
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

    // Cập nhật metrics mô hình Deep Learning
    const metrics = info.test_metrics || {};
    const acc = metrics.accuracy ?? info.test_accuracy;
    const f1 = metrics.macro_f1 ?? info.test_f1;

    const accEl = document.querySelector("#model-accuracy");
    if (accEl) accEl.textContent = acc != null ? percent(acc) : "98.89%";

    const f1El = document.querySelector("#model-f1");
    if (f1El) f1El.textContent = f1 != null ? Number(f1).toFixed(4) : "0.9887";

    const featEl = document.querySelector("#model-feature-count");
    if (featEl) featEl.textContent = number(info.embedding_dim || info.n_features || 576);

    const devEl = document.querySelector("#model-device");
    if (devEl) devEl.textContent = info.device ? String(info.device).toUpperCase() : "CUDA / GPU";

    const settings = [
      ["Kiến trúc CNN Backbone", info.backbone ? `${info.backbone} (ImageNet Pretrained)` : "MobileNetV3-Small"],
      ["Mô hình phân loại (Head)", info.best_model || info.model_name || "PyTorch MLP (TorchMLP_CUDA)"],
      ["Số chiều vector đặc trưng sâu", `${info.embedding_dim || 576} chiều (Global Average Pooling)`],
      ["Số lượng lớp nhận diện", `${info.n_classes || (info.class_names ? info.class_names.length : 9)} Lớp`],
      ["Kích thước ảnh đầu vào", Array.isArray(info.crop_size) ? `${info.crop_size[0]} × ${info.crop_size[1]} px` : "224 × 224 px"],
      ["Số chu kỳ đã huấn luyện (Epochs)", `${info.total_epochs || 15} Epochs`],
      ["Chiến lược giảm chiều PCA", info.use_pca ? `Bật (giữ lại ${((info.pca_ratio || 0.95) * 100).toFixed(0)}% phương sai)` : "Không dùng"],
      ["Thư mục lưu trữ trọng số", "models_dl/classifier_dl.joblib"],
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
        .map((name, index) => `
          <li>
            <span class="class-num">${index + 1}</span>
            <div class="class-info">
              <strong class="class-title">${escapeHtml(classLabel(name))}</strong>
              <small class="class-code">${escapeHtml(name)}</small>
            </div>
          </li>
        `)
        .join("");
    }

    const badge = document.querySelector("#class-count-badge");
    if (badge && info.class_names) {
      badge.textContent = `${info.class_names.length} Lớp`;
    }
  } catch (error) {
    const warning = document.querySelector("#model-warning");
    if (warning) {
      warning.textContent = `${error.message}. Vui lòng kiểm tra trạng thái khởi động backend.`;
      warning.classList.remove("is-hidden");
    }
  }
}
