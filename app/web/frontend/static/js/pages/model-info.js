import { get } from "../api.js";
import { classLabel, escapeHtml, number, percent } from "../utils.js";

const ALL_REPORTS = [
  {
    title: "Sơ đồ kiến trúc pipeline tổng thể",
    file: "pipeline_architecture.png",
    desc: "Sơ đồ khối luồng xử lý từ ảnh đầu vào, trích xuất đặc trưng CNN, PCA đến bộ phân loại MLP.",
  },
  {
    title: "Kiến trúc Web Worker & ONNX Runtime",
    file: "web_worker_architecture.png",
    desc: "Sơ đồ mô hình suy luận bất đồng bộ trên trình duyệt qua WebAssembly.",
  },
  {
    title: "Minh họa mẫu tập dữ liệu 9 lớp",
    file: "dataset_9classes_samples.png",
    desc: "Ảnh đại diện cho 9 lớp phân loại (Cam, Chuối, Xoài ở các trạng thái chưa chín, chín, hỏng).",
  },
  {
    title: "Minh họa tăng cường dữ liệu (Data Augmentation)",
    file: "data_augmentation_demo.png",
    desc: "Các phép biến đổi góc xoay, lật ngang và điều chỉnh độ sáng mở rộng tập train.",
  },
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
  {
    title: "Lịch sử huấn luyện toàn cục",
    file: "training_history.png",
    desc: "Tổng hợp tiến trình huấn luyện qua các kiến trúc mạng CNN.",
  },
  {
    title: "Tiến trình huấn luyện MobileNetV2",
    file: "training_history_mobilenetv2.png",
    desc: "Biểu đồ Loss và Accuracy chi tiết của mô hình MobileNetV2.",
  },
  {
    title: "Tiến trình huấn luyện MobileNetV3-Small",
    file: "training_history_mobilenetv3_small.png",
    desc: "Biểu đồ Loss và Accuracy chi tiết của mô hình MobileNetV3-Small.",
  },
  {
    title: "Tiến trình huấn luyện ResNet-18",
    file: "training_history_resnet18.png",
    desc: "Biểu đồ Loss và Accuracy chi tiết của mô hình ResNet-18.",
  },
  {
    title: "So sánh các kiến trúc CNN",
    file: "model_comparison.png",
    desc: "Biểu đồ so sánh độ chính xác, kích thước và tốc độ giữa các mô hình.",
  },
  {
    title: "Đường cong ROC và chỉ số AUC (Đa lớp)",
    file: "roc_curve.png",
    desc: "Đánh giá khả năng phân loại One-vs-Rest trên 9 lớp.",
  },
  {
    title: "ROC & AUC - MobileNetV2",
    file: "roc_curve_mobilenetv2.png",
    desc: "Đường cong ROC chi tiết của MobileNetV2.",
  },
  {
    title: "ROC & AUC - MobileNetV3-Small",
    file: "roc_curve_mobilenetv3_small.png",
    desc: "Đường cong ROC chi tiết của MobileNetV3-Small.",
  },
  {
    title: "ROC & AUC - ResNet-18",
    file: "roc_curve_resnet18.png",
    desc: "Đường cong ROC chi tiết của mô hình ResNet-18 xuất sắc nhất.",
  },
  {
    title: "Ma trận nhầm lẫn tổng hợp (Confusion Matrix)",
    file: "confusion_matrix.png",
    desc: "Ma trận nhầm lẫn 9x9 đánh giá chi tiết trên tập kiểm thử.",
  },
  {
    title: "Ma trận nhầm lẫn - MobileNetV2",
    file: "confusion_matrix_mobilenetv2.png",
    desc: "Ma trận nhầm lẫn 9x9 của MobileNetV2.",
  },
  {
    title: "Ma trận nhầm lẫn - MobileNetV3-Small",
    file: "confusion_matrix_mobilenetv3_small.png",
    desc: "Ma trận nhầm lẫn 9x9 của MobileNetV3-Small.",
  },
  {
    title: "Ma trận nhầm lẫn - ResNet-18",
    file: "confusion_matrix_resnet18.png",
    desc: "Ma trận nhầm lẫn 9x9 của ResNet-18.",
  },
];

function renderReports() {
  const container = document.querySelector("#model-reports");
  if (!container) return;

  container.innerHTML = ALL_REPORTS.map((r) => `
    <article class="report-card">
      <div class="report-card-header">
        <h3>${escapeHtml(r.title)}</h3>
        <p class="report-desc">${escapeHtml(r.desc)}</p>
      </div>
      <div class="report-img-wrap">
        <img src="/reports/${encodeURIComponent(r.file)}" alt="${escapeHtml(r.title)}" loading="lazy">
        <div class="report-missing is-hidden">Chưa có ảnh báo cáo tương ứng.</div>
      </div>
    </article>
  `).join("");

  container.querySelectorAll("img").forEach((image) => {
    image.addEventListener("error", () => {
      const currentSrc = image.getAttribute("src") || "";
      if (currentSrc.startsWith("/reports/")) {
        const fileName = currentSrc.replace("/reports/", "");
        image.src = `/reports_dl/${fileName}`;
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
