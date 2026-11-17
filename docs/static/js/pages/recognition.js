import { postImage, onModelStatus, switchModel } from "../api.js";
import { classLabel, escapeHtml, percent, ripenessLabel, fruitLabel } from "../utils.js";

const MAX_BYTES = 10 * 1024 * 1024;
const PREDICT_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.8" cy="10.8" r="6.8"/><path d="m16 16 4.5 4.5M10.8 7.5v6.6m-3.3-3.3h6.6"/></svg>';
const READY_BUTTON_HTML = `${PREDICT_ICON} Phân loại ảnh (Học sâu)`;
const REPEAT_BUTTON_HTML = `${PREDICT_ICON} Phân loại lại ảnh`;

const RIPENESS_THEMES = new Map([
  ["Unripe", "unripe"],
  ["Ripe", "ripe"],
  ["Spoiled", "spoiled"],
  ["chin", "ripe"],
  ["chua-chin", "unripe"],
  ["hong", "spoiled"],
]);

let selectedFile = null;
let previewUrl = null;
let pasteListener = null;
let unregisterModelStatus = null;

function feedbackDL(message, isError = true) {
  const element = document.querySelector("#recognition-feedback");
  if (!element) return;
  element.textContent = message;
  element.style.color = isError ? "var(--red)" : "var(--leaf)";
}

function resetResultDL(message = "") {
  const content = document.querySelector("#result-content");
  const empty = document.querySelector("#result-empty");
  const state = document.querySelector("#result-state");
  if (content) content.classList.add("is-hidden");
  if (empty) empty.classList.remove("is-hidden");
  if (state) {
    state.textContent = selectedFile ? "Sẵn sàng phân loại" : "Đang chờ ảnh";
    state.className = "result-state";
  }
  feedbackDL(message);
}

function enableDrop(target) {
  let dragDepth = 0;
  target.addEventListener("dragenter", (event) => {
    event.preventDefault();
    dragDepth += 1;
    target.classList.add("is-dragging");
  });
  target.addEventListener("dragover", (event) => {
    event.preventDefault();
    target.classList.add("is-dragging");
  });
  target.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) target.classList.remove("is-dragging");
  });
  target.addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    target.classList.remove("is-dragging");
    const file = event.dataTransfer?.files?.[0];
    if (file) setFile(file, true, false);
  });
}

function setFile(file, wasDropped = false, wasPasted = false) {
  if (!file) return;

  if (!file.type.startsWith("image/") && !/\.(jpe?g|png|webp|bmp)$/i.test(file.name)) {
    feedbackDL("Vui lòng chọn tệp hình ảnh (JPG, PNG, BMP hoặc WEBP).", true);
    return;
  }

  if (file.size > MAX_BYTES) {
    const hasCurrentFile = Boolean(selectedFile);
    if (!hasCurrentFile) resetResultDL();
    const resultState = document.querySelector("#result-state");
    if (resultState) {
      resultState.textContent = "Tệp quá lớn";
      resultState.className = "result-state is-error";
    }
    feedbackDL(
      hasCurrentFile
        ? "Ảnh mới vượt quá 10 MB. Ảnh đang chọn vẫn được giữ."
        : "Ảnh vượt quá giới hạn 10 MB. Hãy chọn một ảnh nhỏ hơn."
    );
    return;
  }

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  delete document.documentElement.dataset.ripeness;

  const preview = document.querySelector("#image-preview");
  if (preview) {
    preview.src = previewUrl;
    preview.onload = () => {
      const nameEl = document.querySelector("#preview-name");
      const sizeEl = document.querySelector("#preview-size");
      if (nameEl) nameEl.textContent = file.name || "anh-dan-clipboard.png";
      if (sizeEl) {
        sizeEl.textContent = `${(file.size / 1024).toFixed(1)} KB · ${preview.naturalWidth} × ${preview.naturalHeight} px`;
      }
    };
  }

  const dropZone = document.querySelector("#drop-zone");
  const previewWrap = document.querySelector("#image-preview-wrap");
  if (dropZone) dropZone.classList.add("is-hidden");
  if (previewWrap) {
    previewWrap.classList.remove("is-hidden", "is-drop-confirmed", "is-paste-confirmed");
    void previewWrap.offsetWidth;
    if (wasDropped) previewWrap.classList.add("is-drop-confirmed");
    if (wasPasted) previewWrap.classList.add("is-paste-confirmed");
  }

  const button = document.querySelector("#predict-button");
  if (button) {
    button.disabled = false;
    button.innerHTML = READY_BUTTON_HTML;
  }

  resetResultDL(wasPasted ? "Đã dán ảnh từ clipboard thành công." : "");
}

function clearCurrentFile() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  selectedFile = null;
  delete document.documentElement.dataset.ripeness;

  const fileInput = document.querySelector("#image-file");
  if (fileInput) fileInput.value = "";

  const dropZone = document.querySelector("#drop-zone");
  const previewWrap = document.querySelector("#image-preview-wrap");
  if (dropZone) dropZone.classList.remove("is-hidden");
  if (previewWrap) previewWrap.classList.add("is-hidden");

  const button = document.querySelector("#predict-button");
  if (button) {
    button.disabled = true;
    button.innerHTML = READY_BUTTON_HTML;
  }

  resetResultDL();
}

function showPredictionDL(data) {
  const theme = RIPENESS_THEMES.get(data.ripeness) || RIPENESS_THEMES.get(String(data.class_name).split("-")[1]);
  if (theme) document.documentElement.dataset.ripeness = theme;
  else delete document.documentElement.dataset.ripeness;

  const emptyEl = document.querySelector("#result-empty");
  const contentEl = document.querySelector("#result-content");
  const stateEl = document.querySelector("#result-state");
  if (emptyEl) emptyEl.classList.add("is-hidden");
  if (contentEl) contentEl.classList.remove("is-hidden");
  if (stateEl) {
    stateEl.textContent = "Đã phân loại";
    stateEl.className = "result-state is-success";
  }

  const predictedClass = data.class_name || [data.fruit_type, data.ripeness].filter(Boolean).join("-");
  const fruitEl = document.querySelector("#result-fruit");
  const classEl = document.querySelector("#result-class");
  const fruitTypeEl = document.querySelector("#result-fruit-type");
  const ripenessEl = document.querySelector("#result-ripeness");
  const qualityEl = document.querySelector("#result-quality");
  const confidenceEl = document.querySelector("#result-confidence");
  const recEl = document.querySelector("#result-recommendation");
  const timeEl = document.querySelector("#result-time");

  if (fruitEl) fruitEl.textContent = data.class_name_vi || classLabel(predictedClass);
  if (classEl) classEl.textContent = `Lớp mô hình: ${predictedClass}`;
  if (fruitTypeEl) fruitTypeEl.textContent = data.fruit_type_vi || fruitLabel(data.fruit_type);
  if (ripenessEl) ripenessEl.textContent = data.ripeness_vi || (data.ripeness ? ripenessLabel(data.ripeness) : "Đã xác định");
  if (qualityEl) {
    qualityEl.textContent = data.quality || "Đạt chuẩn";
    qualityEl.className = `diag-val quality-${data.quality_code || "good"}`;
  }
  if (confidenceEl) {
    confidenceEl.textContent = data.confidence == null ? "Không có" : percent(data.confidence);
  }
  if (recEl) {
    recEl.textContent = data.recommendation || "Sản phẩm được đánh giá qua mô hình mạng nơ-ron tích chập.";
  }
  if (timeEl) {
    timeEl.textContent = `${Number(data.processing_time_ms || 0).toFixed(1)} ms`;
  }

  const modelNameEl = document.querySelector("#result-model-name");
  if (modelNameEl) {
    const select = document.querySelector("#model-select");
    const currentModelText = select ? select.options[select.selectedIndex].text.split("(")[0].trim() : "ResNet-18";
    modelNameEl.textContent = `${currentModelText} (Wasm · ${Number(data.latency_ms || data.processing_time_ms || 0)}ms)`;
  }

  // Hiển thị thanh xác suất 9 lớp
  const probabilityBlock = document.querySelector("#probability-block");
  const probabilities = data.probabilities;
  if (!probabilities || !Object.keys(probabilities).length) {
    if (probabilityBlock) probabilityBlock.classList.add("is-hidden");
  } else {
    if (probabilityBlock) probabilityBlock.classList.remove("is-hidden");
    const listEl = document.querySelector("#probability-list");
    if (listEl) {
      listEl.innerHTML = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1])
        .map(([name, value], index) => {
          const isTop = index === 0;
          const percentage = Math.max(0, Math.min(100, Number(value) * 100));
          return `
            <div class="probability-row ${isTop ? "is-top-prob" : ""}">
              <span class="prob-name" title="${escapeHtml(name)}">${escapeHtml(classLabel(name))}</span>
              <span class="bar-track">
                <span class="bar-fill" style="display:block;width:${percentage}%"></span>
              </span>
              <strong class="prob-val">${percent(value)}</strong>
            </div>
          `;
        }).join("");
    }
  }

  feedbackDL("Kết quả đã được tính toán và lưu vào lịch sử.", false);
}

// Bắt sự kiện dán ảnh từ clipboard (Ctrl + V)
function handleGlobalPaste(event) {
  const items = event.clipboardData?.items;
  if (!items || !items.length) return;

  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const blob = item.getAsFile();
      if (blob) {
        event.preventDefault();
        const ext = blob.type.split("/")[1] || "png";
        const pasteFile = new File([blob], `clipboard-${Date.now()}.${ext}`, { type: blob.type });
        setFile(pasteFile, false, true);
        feedbackDL("Đã dán ảnh từ bộ nhớ tạm (Ctrl + V).", false);
        return;
      }
    }
  }
}

// Nạp danh sách ảnh mẫu chọn nhanh (Quick-select Fruit Gallery)
async function loadSampleGallery() {
  const container = document.querySelector("#sample-gallery-grid");
  if (!container) return;

  try {
    let res = await fetch("/static/data/samples.json");
    if (!res.ok) res = await fetch("static/data/samples.json");
    if (!res.ok) return;
    const samples = await res.json();

    container.innerHTML = samples.map((s) => `
      <button type="button" class="sample-item" data-id="${escapeHtml(s.id)}" data-name="${escapeHtml(s.name)}" data-img="${escapeHtml(s.image)}" title="Thử nhanh: ${escapeHtml(s.name)}">
        <div class="sample-thumb-wrap">
          <img src="${escapeHtml(s.image)}" alt="${escapeHtml(s.name)}" loading="lazy">
          <span class="sample-badge">${escapeHtml(s.category)}</span>
        </div>
        <span class="sample-name">${escapeHtml(s.name)}</span>
      </button>
    `).join("");

    container.querySelectorAll(".sample-item").forEach((btn) => {
      btn.addEventListener("click", async () => {
        container.querySelectorAll(".sample-item").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");

        const imgPath = btn.dataset.img;
        const sampleName = btn.dataset.name;

        feedbackDL(`Đang nạp ảnh mẫu "${sampleName}"...`, false);
        try {
          let imgRes = await fetch(imgPath.startsWith("/") ? imgPath : `/${imgPath}`);
          if (!imgRes.ok) imgRes = await fetch(imgPath);
          const blob = await imgRes.blob();
          const file = new File([blob], `${btn.dataset.id}.jpg`, { type: "image/jpeg" });
          setFile(file, false, false);

          // Tự động phân loại ngay lập tức khi click ảnh mẫu
          const predictButton = document.querySelector("#predict-button");
          if (predictButton) {
            predictButton.click();
          }
        } catch (err) {
          feedbackDL(`Không thể nạp ảnh mẫu: ${err.message}`, true);
        }
      });
    });

    // Hỗ trợ lăn chuột dọc để cuộn ngang danh sách ảnh mẫu trên Desktop
    container.addEventListener("wheel", (e) => {
      if (e.deltaY !== 0) {
        e.preventDefault();
        container.scrollLeft += e.deltaY;
      }
    }, { passive: false });
  } catch (err) {
    console.warn("Không thể tải danh sách ảnh mẫu:", err);
  }
}

// Thiết lập bộ chọn chuyển đổi kiến trúc mô hình ONNX
function setupModelSwitcher() {
  const select = document.querySelector("#model-select");
  if (!select) return;

  select.addEventListener("change", async () => {
    const targetModelFile = select.value;
    const modelText = select.options[select.selectedIndex].text;
    feedbackDL(`Đang chuyển sang mô hình ${modelText}...`, false);

    const stateEl = document.querySelector("#result-state");
    if (stateEl) {
      stateEl.textContent = "Đang đổi mô hình...";
      stateEl.className = "result-state is-running";
    }

    try {
      await switchModel(targetModelFile);
      feedbackDL(`Đã kích hoạt mô hình ${modelText}!`, false);
      if (stateEl) {
        stateEl.textContent = "Mô hình sẵn sàng";
        stateEl.className = "result-state is-success";
      }

      const modelNameEl = document.querySelector("#result-model-name");
      if (modelNameEl) {
        modelNameEl.textContent = `${modelText.split("(")[0].trim()} (Wasm)`;
      }

      // Nếu đã có ảnh đang xem trước, tự động suy luận lại trên mô hình mới
      const predictBtn = document.querySelector("#predict-button");
      if (predictBtn && !predictBtn.disabled && selectedFile) {
        predictBtn.click();
      }
    } catch (err) {
      feedbackDL(`Lỗi chuyển đổi mô hình: ${err.message}`, true);
    }
  });
}

export function mount() {
  const fileInput = document.querySelector("#image-file");
  const dropZone = document.querySelector("#drop-zone");
  const previewWrap = document.querySelector("#image-preview-wrap");
  const clearButton = document.querySelector("#clear-image");
  const pasteButton = document.querySelector("#paste-button");
  const predictButton = document.querySelector("#predict-button");

  // Kích hoạt bộ chọn mô hình và danh sách ảnh mẫu
  setupModelSwitcher();
  loadSampleGallery();

  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      setFile(e.target.files?.[0]);
    });
  }

  if (dropZone) enableDrop(dropZone);
  if (previewWrap) enableDrop(previewWrap);

  if (clearButton) {
    clearButton.addEventListener("click", clearCurrentFile);
  }

  if (pasteButton) {
    pasteButton.addEventListener("click", async () => {
      try {
        if (navigator.clipboard?.read) {
          const items = await navigator.clipboard.read();
          for (const item of items) {
            for (const type of item.types) {
              if (type.startsWith("image/")) {
                const blob = await item.getType(type);
                const ext = type.split("/")[1] || "png";
                const pasteFile = new File([blob], `clipboard-${Date.now()}.${ext}`, { type });
                setFile(pasteFile, false, true);
                feedbackDL("Đã dán ảnh từ bộ nhớ tạm (Ctrl + V).", false);
                return;
              }
            }
          }
        }
        feedbackDL("Hãy nhấn Ctrl + V để dán ảnh đã copy từ clipboard.", false);
      } catch (err) {
        feedbackDL("Hãy nhấn phím Ctrl + V để dán ảnh từ bộ nhớ tạm.", false);
      }
    });
  }

  // Đăng ký listener Ctrl + V toàn cục
  if (!pasteListener) {
    pasteListener = handleGlobalPaste;
    window.addEventListener("paste", pasteListener);
  }

  // Lắng nghe trạng thái nạp mô hình ONNX
  unregisterModelStatus = onModelStatus(({ status, message }) => {
    const stateEl = document.querySelector("#result-state");
    if (!stateEl || selectedFile) return;

    if (status === "loading_model") {
      stateEl.textContent = "Đang nạp ONNX (~4.2MB)...";
      stateEl.className = "result-state is-running";
    } else if (status === "ready") {
      stateEl.textContent = "Mô hình sẵn sàng (Wasm)";
      stateEl.className = "result-state is-success";
    } else if (status === "error") {
      stateEl.textContent = "Lỗi nạp mô hình";
      stateEl.className = "result-state is-error";
      feedbackDL(message || "Không thể tải mô hình ONNX.", true);
    }
  });

  if (predictButton) {
    predictButton.addEventListener("click", async () => {
      if (!selectedFile) return;
      predictButton.disabled = true;
      predictButton.innerHTML = '<span class="spinner" aria-hidden="true"></span> Đang trích xuất đặc trưng & phân loại…';
      const stateEl = document.querySelector("#result-state");
      if (stateEl) {
        stateEl.textContent = "Đang xử lý...";
        stateEl.className = "result-state is-running";
      }
      feedbackDL("");

      try {
        const data = await postImage(selectedFile);
        showPredictionDL(data);
        predictButton.innerHTML = REPEAT_BUTTON_HTML;
      } catch (error) {
        const errorState = document.querySelector("#result-state");
        if (errorState) {
          errorState.textContent = "Lỗi xử lý";
          errorState.className = "result-state is-error";
        }
        feedbackDL(error.message || "Không thể phân loại ảnh.");
        predictButton.innerHTML = READY_BUTTON_HTML;
      } finally {
        predictButton.disabled = false;
      }
    });
  }
}

export function unmount() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  selectedFile = null;
  if (pasteListener) {
    window.removeEventListener("paste", pasteListener);
    pasteListener = null;
  }
  if (unregisterModelStatus) {
    unregisterModelStatus();
    unregisterModelStatus = null;
  }
}
