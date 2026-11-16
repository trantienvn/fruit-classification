import { postImage, postTabular } from "../api.js";
import { classLabel, escapeHtml, percent, ripenessLabel } from "../utils.js";

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

// =========================================================================
// PHẦN 1: BẢN HỌC SÂU (DEEP LEARNING - ẢNH)
// =========================================================================

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
    setFile(event.dataTransfer?.files?.[0], true);
  });
}

function setFile(file, wasDropped = false) {
  if (!file) return;
  if (file.size > MAX_BYTES) {
    const hasCurrentFile = Boolean(selectedFile);
    if (!hasCurrentFile) resetResultDL();
    const resultState = document.querySelector("#result-state");
    resultState.textContent = "Tệp quá lớn";
    resultState.className = "result-state is-error";
    feedbackDL(
      hasCurrentFile
        ? "Ảnh thay thế vượt quá 10 MB. Ảnh đang chọn vẫn được giữ."
        : "Ảnh vượt quá 10 MB. Hãy chọn một tệp nhỏ hơn."
    );
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  delete document.documentElement.dataset.ripeness;
  const preview = document.querySelector("#image-preview");
  preview.src = previewUrl;
  preview.onload = () => {
    document.querySelector("#preview-name").textContent = file.name;
    document.querySelector("#preview-size").textContent = `${(file.size / 1024).toFixed(1)} KB · ${preview.naturalWidth} × ${preview.naturalHeight} px`;
  };
  document.querySelector("#drop-zone").classList.add("is-hidden");
  const previewWrap = document.querySelector("#image-preview-wrap");
  previewWrap.classList.remove("is-hidden", "is-drop-confirmed");
  if (wasDropped) {
    void previewWrap.offsetWidth;
    previewWrap.classList.add("is-drop-confirmed");
  }
  const button = document.querySelector("#predict-button");
  button.disabled = Boolean(button.querySelector(".spinner"));
  resetResultDL();
}

function showPredictionDL(data) {
  const theme = RIPENESS_THEMES.get(data.ripeness) || RIPENESS_THEMES.get(String(data.class_name).split("-")[1]);
  if (theme) document.documentElement.dataset.ripeness = theme;
  else delete document.documentElement.dataset.ripeness;

  document.querySelector("#result-empty").classList.add("is-hidden");
  document.querySelector("#result-content").classList.remove("is-hidden");
  document.querySelector("#result-state").textContent = "Đã phân loại";
  document.querySelector("#result-state").className = "result-state is-success";

  const predictedClass = data.class_name || [data.fruit_type, data.ripeness].filter(Boolean).join("-");
  document.querySelector("#result-fruit").textContent = data.class_name_vi || classLabel(predictedClass);
  document.querySelector("#result-class").textContent = `Lớp mô hình: ${predictedClass}`;
  document.querySelector("#result-ripeness").textContent = data.ripeness ? ripenessLabel(data.ripeness) : "Đã xác định";
  document.querySelector("#result-confidence").textContent = data.confidence == null ? "Không có" : percent(data.confidence);
  document.querySelector("#result-time").textContent = `${Number(data.processing_time_ms || 0).toFixed(1)} ms`;

  const probabilityBlock = document.querySelector("#probability-block");
  const probabilities = data.probabilities;
  if (!probabilities || !Object.keys(probabilities).length) {
    probabilityBlock.classList.add("is-hidden");
  } else {
    probabilityBlock.classList.remove("is-hidden");
    document.querySelector("#probability-list").innerHTML = Object.entries(probabilities)
      .sort((a, b) => b[1] - a[1])
      .map(([name, value]) => `
        <div class="probability-row">
          <span title="${escapeHtml(name)}">${escapeHtml(classLabel(name))}</span>
          <span class="bar-track">
            <span class="bar-fill" style="display:block;width:${Math.max(0, Math.min(100, Number(value) * 100))}%"></span>
          </span>
          <strong>${percent(value)}</strong>
        </div>
      `).join("");
  }
  feedbackDL("Kết quả đã được lưu vào lịch sử.", false);
}

// =========================================================================
// PHẦN 2: BẢN HỌC MÁY BẢNG SỐ LIỆU (TABULAR ML)
// =========================================================================

const TABULAR_FEATURES = [
  "size",
  "weight",
  "sweetness",
  "crunchiness",
  "juiciness",
  "acidity",
  "ripeness",
];

const PRESETS = {
  good: {
    size: 1.2,
    weight: 1.5,
    sweetness: 2.1,
    crunchiness: 1.4,
    juiciness: 1.8,
    acidity: -0.8,
    ripeness: 1.1,
  },
  bad: {
    size: -1.2,
    weight: -1.0,
    sweetness: -1.8,
    crunchiness: -1.5,
    juiciness: -1.2,
    acidity: 2.3,
    ripeness: -1.4,
  },
  reset: {
    size: 0.0,
    weight: 0.0,
    sweetness: 0.0,
    crunchiness: 0.0,
    juiciness: 0.0,
    acidity: 0.0,
    ripeness: 0.0,
  },
};

function feedbackTabular(message, isError = true) {
  const element = document.querySelector("#tabular-feedback");
  if (!element) return;
  element.textContent = message;
  element.style.color = isError ? "var(--red)" : "var(--leaf)";
}

function getTabularValues() {
  const values = {};
  for (const feat of TABULAR_FEATURES) {
    const range = document.querySelector(`#range-${feat}`);
    values[feat] = range ? parseFloat(range.value) : 0.0;
  }
  return values;
}

function setTabularValues(presetValues) {
  for (const [feat, val] of Object.entries(presetValues)) {
    const range = document.querySelector(`#range-${feat}`);
    const display = document.querySelector(`#val-${feat}`);
    const numInput = document.querySelector(`#input-${feat}`);
    if (range) range.value = val;
    if (display) display.textContent = Number(val).toFixed(2);
    if (numInput) numInput.value = val;
  }
}

function initTabularSliders() {
  for (const feat of TABULAR_FEATURES) {
    const range = document.querySelector(`#range-${feat}`);
    const display = document.querySelector(`#val-${feat}`);
    const numInput = document.querySelector(`#input-${feat}`);
    if (!range) continue;

    range.addEventListener("input", () => {
      const val = parseFloat(range.value);
      if (display) display.textContent = val.toFixed(2);
      if (numInput) numInput.value = val;
    });
  }

  document.querySelector("#preset-good")?.addEventListener("click", () => {
    setTabularValues(PRESETS.good);
  });
  document.querySelector("#preset-bad")?.addEventListener("click", () => {
    setTabularValues(PRESETS.bad);
  });
  document.querySelector("#preset-reset")?.addEventListener("click", () => {
    setTabularValues(PRESETS.reset);
  });
}

function showTabularPrediction(data) {
  document.querySelector("#tabular-result-empty").classList.add("is-hidden");
  document.querySelector("#tabular-result-content").classList.remove("is-hidden");
  document.querySelector("#tabular-result-state").textContent = "Đã đánh giá";
  document.querySelector("#tabular-result-state").className = "result-state is-success";

  const isGood = data.predicted_class === "Good";
  const badge = document.querySelector("#tabular-quality-badge");
  badge.textContent = isGood ? "TỐT (ĐẠT CHUẨN XUẤT KHẨU)" : "KÉM (CHƯA ĐẠT CHUẨN)";
  badge.className = `tabular-quality-badge ${isGood ? "is-good" : "is-bad"}`;

  document.querySelector("#tabular-model-name").textContent = `Mô hình: ${data.model_name || "SVM_RBF (Học máy bảng)"}`;
  document.querySelector("#tabular-confidence").textContent = percent(data.confidence);

  const probGood = data.probabilities?.Good ?? 0;
  const probBad = data.probabilities?.Bad ?? 0;
  document.querySelector("#tabular-prob-good").textContent = percent(probGood);
  document.querySelector("#tabular-prob-bad").textContent = percent(probBad);

  const assessmentList = document.querySelector("#tabular-assessment-list");
  if (assessmentList && data.assessments) {
    assessmentList.innerHTML = data.assessments
      .map(
        (a) => `
        <div class="assessment-row">
          <div class="assessment-feat">
            <strong>${escapeHtml(a.feature_vi)}</strong>
            <span class="feat-en">(${escapeHtml(a.feature)})</span>
          </div>
          <span class="assessment-val">${Number(a.value).toFixed(2)}</span>
          <span class="assessment-status">${escapeHtml(a.status)}</span>
        </div>
      `
      )
      .join("");
  }

  feedbackTabular("Đã tính toán phân loại theo bảng số liệu thành công.", false);
}

// =========================================================================
// PHẦN 3: MOUNT & CHUYỂN TAB
// =========================================================================

export function mount() {
  // 1. Tab Switching
  const btnDL = document.querySelector("#tab-btn-dl");
  const btnTabular = document.querySelector("#tab-btn-tabular");
  const viewDL = document.querySelector("#view-dl");
  const viewTabular = document.querySelector("#view-tabular");

  btnDL?.addEventListener("click", () => {
    btnDL.classList.add("is-active");
    btnDL.setAttribute("aria-selected", "true");
    btnTabular?.classList.remove("is-active");
    btnTabular?.setAttribute("aria-selected", "false");
    viewDL?.classList.remove("is-hidden");
    viewTabular?.classList.add("is-hidden");
  });

  btnTabular?.addEventListener("click", () => {
    btnTabular.classList.add("is-active");
    btnTabular.setAttribute("aria-selected", "true");
    btnDL?.classList.remove("is-active");
    btnDL?.setAttribute("aria-selected", "false");
    viewTabular?.classList.remove("is-hidden");
    viewDL?.classList.add("is-hidden");
  });

  // 2. Gắn kết bản Học Sâu (Deep Learning)
  const input = document.querySelector("#image-file");
  const dropZone = document.querySelector("#drop-zone");
  const inputPanel = document.querySelector(".recognition-input");
  const previewWrap = document.querySelector("#image-preview-wrap");
  const button = document.querySelector("#predict-button");

  if (previewWrap) {
    previewWrap.addEventListener("animationend", (event) => {
      if (event.animationName === "drop-confirm") previewWrap.classList.remove("is-drop-confirmed");
    });
  }

  if (input) {
    input.addEventListener("change", () => {
      setFile(input.files?.[0]);
      input.value = "";
    });
  }

  if (inputPanel) {
    enableDrop(inputPanel);
  }

  document.querySelector("#clear-image")?.addEventListener("click", () => {
    selectedFile = null;
    if (input) input.value = "";
    delete document.documentElement.dataset.ripeness;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    previewWrap?.classList.add("is-hidden");
    dropZone?.classList.remove("is-hidden");
    if (button) button.disabled = true;
    resetResultDL();
  });

  button?.addEventListener("click", async () => {
    if (!selectedFile) return;
    const requestFile = selectedFile;
    button.disabled = true;
    button.innerHTML = '<span class="spinner" aria-hidden="true"></span> Đang phân loại';
    document.querySelector("#result-state").textContent = "Đang xử lý ảnh";
    feedbackDL("");
    try {
      const result = await postImage(requestFile);
      if (selectedFile === requestFile) showPredictionDL(result);
    } catch (error) {
      if (selectedFile !== requestFile) return;
      document.querySelector("#result-state").textContent = "Không hoàn tất";
      document.querySelector("#result-state").className = "result-state is-error";
      feedbackDL(error.message);
    } finally {
      const selectionChanged = selectedFile !== requestFile;
      button.disabled = selectionChanged ? !selectedFile : false;
      button.innerHTML = selectionChanged ? READY_BUTTON_HTML : REPEAT_BUTTON_HTML;
    }
  });

  // 3. Gắn kết bản Học Máy Bảng Số Liệu (Tabular ML)
  initTabularSliders();

  const tabularButton = document.querySelector("#tabular-predict-button");
  tabularButton?.addEventListener("click", async () => {
    const values = getTabularValues();
    tabularButton.disabled = true;
    tabularButton.innerHTML = '<span class="spinner" aria-hidden="true"></span> Đang đánh giá';
    document.querySelector("#tabular-result-state").textContent = "Đang tính toán...";
    feedbackTabular("");

    try {
      const result = await postTabular(values);
      showTabularPrediction(result);
    } catch (error) {
      document.querySelector("#tabular-result-state").textContent = "Lỗi tính toán";
      document.querySelector("#tabular-result-state").className = "result-state is-error";
      feedbackTabular(error.message);
    } finally {
      tabularButton.disabled = false;
      tabularButton.innerHTML = `
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
        Đánh giá chất lượng (Học máy)
      `;
    }
  });
}
