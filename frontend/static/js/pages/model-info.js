import { get } from "../api.js";
import { classLabel, escapeHtml, number, percent } from "../utils.js";

const ALL_REPORTS = [
  {
    title: "1. Sơ đồ kiến trúc pipeline tổng thể",
    file: "pipeline_architecture.png",
    desc: "Sơ đồ khối luồng xử lý từ ảnh đầu vào, trích xuất đặc trưng CNN, PCA đến bộ phân loại MLP.",
  },
  {
    title: "2. Kiến trúc Web Worker & ONNX Runtime",
    file: "web_worker_architecture.png",
    desc: "Sơ đồ mô hình suy luận bất đồng bộ trên trình duyệt qua WebAssembly.",
  },
  {
    title: "3. Minh họa mẫu tập dữ liệu 9 lớp",
    file: "dataset_9classes_samples.png",
    desc: "Ảnh đại diện cho 9 lớp phân loại (Cam, Chuối, Xoài ở các trạng thái chưa chín, chín, hỏng).",
  },
  {
    title: "4. Minh họa tăng cường dữ liệu (Data Augmentation)",
    file: "data_augmentation_demo.png",
    desc: "Các phép biến đổi góc xoay, lật ngang và điều chỉnh độ sáng mở rộng tập train.",
  },
  {
    title: "5. Sơ đồ xu hướng Loss theo Epoch (Train vs Val)",
    file: "loss_curve.png",
    desc: "Đường cong Cross-Entropy Loss qua các Epoch, kiểm soát hiện tượng Overfitting/Underfitting.",
  },
  {
    title: "6. Tiến trình huấn luyện & Tốc độ hội tụ (Loss & Accuracy)",
    file: "epoch_training.png",
    desc: "Đánh giá chi tiết sự tương quan giữa hàm mất mát và độ chính xác phân loại theo từng chu kỳ.",
  },
  {
    title: "7. Lịch sử huấn luyện toàn cục",
    file: "training_history.png",
    desc: "Tổng hợp tiến trình huấn luyện qua các kiến trúc mạng CNN.",
  },
  {
    title: "8. Tiến trình huấn luyện MobileNetV2",
    file: "training_history_mobilenetv2.png",
    desc: "Biểu đồ Loss và Accuracy chi tiết của mô hình MobileNetV2.",
  },
  {
    title: "9. Tiến trình huấn luyện MobileNetV3-Small",
    file: "training_history_mobilenetv3_small.png",
    desc: "Biểu đồ Loss và Accuracy chi tiết của mô hình MobileNetV3-Small.",
  },
  {
    title: "10. Tiến trình huấn luyện ResNet-18",
    file: "training_history_resnet18.png",
    desc: "Biểu đồ Loss và Accuracy chi tiết của mô hình ResNet-18.",
  },
  {
    title: "11. So sánh các kiến trúc CNN",
    file: "model_comparison.png",
    desc: "Biểu đồ so sánh độ chính xác, kích thước và tốc độ giữa các mô hình.",
  },
  {
    title: "12. Đường cong ROC và chỉ số AUC (Đa lớp)",
    file: "roc_curve.png",
    desc: "Đánh giá khả năng phân loại One-vs-Rest trên 9 lớp.",
  },
  {
    title: "13. ROC & AUC - MobileNetV2",
    file: "roc_curve_mobilenetv2.png",
    desc: "Đường cong ROC chi tiết của MobileNetV2.",
  },
  {
    title: "14. ROC & AUC - MobileNetV3-Small",
    file: "roc_curve_mobilenetv3_small.png",
    desc: "Đường cong ROC chi tiết của MobileNetV3-Small.",
  },
  {
    title: "15. ROC & AUC - ResNet-18",
    file: "roc_curve_resnet18.png",
    desc: "Đường cong ROC chi tiết của mô hình ResNet-18 xuất sắc nhất.",
  },
  {
    title: "16. Ma trận nhầm lẫn tổng hợp (Confusion Matrix)",
    file: "confusion_matrix.png",
    desc: "Đánh giá chi tiết các trường hợp dự đoán đúng và các ca nhầm lẫn giữa 9 lớp quả - độ chín - hư hỏng trên 270 ảnh tập kiểm thử độc lập.",
  },
  {
    title: "17. Ma trận nhầm lẫn - MobileNetV2",
    file: "confusion_matrix_mobilenetv2.png",
    desc: "Ma trận nhầm lẫn 9x9 của MobileNetV2.",
  },
  {
    title: "18. Ma trận nhầm lẫn - MobileNetV3-Small",
    file: "confusion_matrix_mobilenetv3_small.png",
    desc: "Ma trận nhầm lẫn 9x9 của MobileNetV3-Small.",
  },
  {
    title: "19. Ma trận nhầm lẫn - ResNet-18",
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
        <img src="static/img/${encodeURIComponent(r.file)}" alt="${escapeHtml(r.title)}" loading="lazy">
        <div class="report-missing is-hidden">Chưa có ảnh báo cáo tương ứng.</div>
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
