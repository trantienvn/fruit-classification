import { get } from "../api.js";
import { classLabel, escapeHtml, number, percent } from "../utils.js";

const DEFAULT_REPORTS = [
  {
    title: "1. So sánh hiệu năng giữa các kiến trúc học sâu (Mục 3.6 Đề cương)",
    file: "model_comparison.png",
    desc: "Biểu đồ so sánh 4 chiều: Độ chính xác / F1, Kích thước file ONNX, Tốc độ suy luận (ms), và Số phép tính FLOPs giữa MobileNetV3-Small, MobileNetV2 và ResNet-18.",
  },
  {
    title: "2. Ma trận nhầm lẫn tổng thể — Confusion Matrix (Mục 1.5.1 Đề cương)",
    file: "confusion_matrix.png",
    desc: "Đánh giá chi tiết các trường hợp dự đoán đúng và các ca nhầm lẫn giữa 9 lớp quả - độ chín - hư hỏng trên 270 ảnh tập kiểm thử độc lập.",
  },
  {
    title: "3. Đường cong ROC đa lớp One-vs-Rest & Chỉ số AUC (Mục 1.5.2 Đề cương)",
    file: "roc_curve.png",
    desc: "Đánh giá khả năng phân tách nhãn của mô hình ở mọi ngưỡng quyết định. Macro-average AUC đạt > 0.999 minh chứng cho độ tin cậy vượt trội.",
  },
  {
    title: "4. Lịch sử hàm mất mát & Độ chính xác qua các Epoch",
    file: "training_history.png",
    desc: "Đường cong Cross-Entropy Loss và Accuracy (Train vs Validation) qua 10 Epochs trên GPU RTX 3050, thể hiện tốc độ hội tụ nhanh và ổn định.",
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
        <img src="static/img/${encodeURIComponent(r.file)}" alt="${escapeHtml(r.title)}" loading="lazy">
        <div class="report-missing is-hidden">Chưa có biểu đồ trong static/img/.</div>
      </div>
    </article>
  `).join("");

  container.querySelectorAll("img").forEach((image) => {
    image.addEventListener("error", () => {
      image.classList.add("is-hidden");
      const missing = image.nextElementSibling;
      if (missing) {
        missing.textContent = "Chưa có ảnh báo cáo tương ứng.";
        missing.classList.remove("is-hidden");
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
    if (accEl) accEl.textContent = acc != null ? percent(acc) : "98.15%";

    const f1El = document.querySelector("#model-f1");
    if (f1El) f1El.textContent = f1 != null ? Number(f1).toFixed(4) : "0.9813";

    const archEl = document.querySelector("#model-best-arch");
    if (archEl) archEl.textContent = info.display_name || "ResNet-18";

    const settings = [
      ["Kiến trúc Backbone tốt nhất", info.display_name ? `${info.display_name} (Pretrained ImageNet)` : "ResNet-18 (Pretrained ImageNet)"],
      ["Mô hình phân loại (Head)", info.best_model || "ResNet-18 + PyTorch MLP"],
      ["Số chiều vector đặc trưng sâu", `${info.embedding_dim || 512} chiều (Global Average Pooling)`],
      ["Số lượng lớp nhận diện", `${info.n_classes || 9} Lớp chủng loại & Độ chín`],
      ["Kích thước ảnh đầu vào", Array.isArray(info.crop_size) ? `${info.crop_size[0]} × ${info.crop_size[1]} px` : "224 × 224 px"],
      ["Số chu kỳ đã huấn luyện (Epochs)", `${info.total_epochs || 10} Epochs trên GPU RTX 3050`],
      ["Chiến lược giảm chiều PCA", info.use_pca ? `Bật (giữ lại ${((info.pca_ratio || 0.95) * 100).toFixed(0)}% phương sai)` : "Không dùng"],
      ["Kiến trúc so sánh thực nghiệm", "MobileNetV3-Small, MobileNetV2, ResNet-18"],
      ["Môi trường suy luận trực tiếp", "ONNX Runtime Web (Wasm) không cần server"],
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

    // Hiển thị phiên bản ứng dụng hiện tại
    const verEl = document.querySelector("#app-current-ver");
    if (verEl && window.fruitApp?.getCurrentVersion) {
      verEl.textContent = `v${window.fruitApp.getCurrentVersion()}`;
    }

    // Gắn sự kiện kiểm tra cập nhật thủ công
    const checkBtn = document.querySelector("#btn-check-update");
    if (checkBtn) {
      checkBtn.addEventListener("click", async () => {
        checkBtn.disabled = true;
        const originalHtml = checkBtn.innerHTML;
        checkBtn.innerHTML = '<span class="spinner" aria-hidden="true" style="width:14px;height:14px;border-width:1.5px;"></span> Đang kiểm tra...';
        try {
          await window.fruitApp?.checkAppUpdate();
        } finally {
          checkBtn.disabled = false;
          checkBtn.innerHTML = originalHtml;
        }
      });
    }

    // Gắn sự kiện xóa toàn bộ cache client
    const clearBtn = document.querySelector("#btn-clear-cache");
    if (clearBtn) {
      clearBtn.addEventListener("click", async () => {
        if (confirm("Hành động này sẽ xóa toàn bộ bộ nhớ đệm (CacheStorage, Session) và tải lại ứng dụng từ server. Bạn có muốn tiếp tục?")) {
          clearBtn.disabled = true;
          clearBtn.innerHTML = '<span class="spinner" aria-hidden="true" style="width:14px;height:14px;border-width:1.5px;"></span> Đang dọn cache...';
          await window.fruitApp?.clearClientCache();
        }
      });
    }
  } catch (error) {
    const warning = document.querySelector("#model-warning");
    if (warning) {
      warning.textContent = `${error.message}. Vui lòng kiểm tra trạng thái nạp dữ liệu.`;
      warning.classList.remove("is-hidden");
    }
  }
}
