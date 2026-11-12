import { postImage } from "../api.js";
import { classLabel, escapeHtml, percent, ripenessLabel } from "../utils.js";

const MAX_BYTES = 10 * 1024 * 1024;
const PREDICT_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.8" cy="10.8" r="6.8"/><path d="m16 16 4.5 4.5M10.8 7.5v6.6m-3.3-3.3h6.6"/></svg>';
const READY_BUTTON_HTML = `${PREDICT_ICON} Phân loại ảnh`;
const REPEAT_BUTTON_HTML = `${PREDICT_ICON} Phân loại lại ảnh`;
const RIPENESS_THEMES = new Map([
  ["Unripe", "unripe"],
  ["Ripe", "ripe"],
  ["Spoiled", "spoiled"],
]);
let selectedFile = null;
let previewUrl = null;

function feedback(message, isError = true) {
  const element = document.querySelector("#recognition-feedback");
  element.textContent = message;
  element.style.color = isError ? "var(--red)" : "var(--leaf)";
}

function resetResult(message = "") {
  document.querySelector("#result-content").classList.add("is-hidden");
  document.querySelector("#result-empty").classList.remove("is-hidden");
  document.querySelector("#result-state").textContent = selectedFile ? "Sẵn sàng phân loại" : "Đang chờ ảnh";
  document.querySelector("#result-state").className = "result-state";
  feedback(message);
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
    if (!hasCurrentFile) resetResult();
    const resultState = document.querySelector("#result-state");
    resultState.textContent = "Tệp quá lớn";
    resultState.className = "result-state is-error";
    feedback(hasCurrentFile
      ? "Ảnh thay thế vượt quá 10 MB. Ảnh đang chọn vẫn được giữ."
      : "Ảnh vượt quá 10 MB. Hãy chọn một tệp nhỏ hơn.");
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
  resetResult();
}

function showPrediction(data) {
  const theme = RIPENESS_THEMES.get(data.ripeness);
  if (theme) document.documentElement.dataset.ripeness = theme;
  else delete document.documentElement.dataset.ripeness;
  document.querySelector("#result-empty").classList.add("is-hidden");
  document.querySelector("#result-content").classList.remove("is-hidden");
  document.querySelector("#result-state").textContent = "Đã phân loại";
  document.querySelector("#result-state").className = "result-state is-success";
  const predictedClass = data.class_name || [data.fruit_type, data.ripeness].filter(Boolean).join(" ");
  document.querySelector("#result-fruit").textContent = data.class_name_vi || classLabel(predictedClass);
  document.querySelector("#result-class").textContent = `Lớp mô hình: ${predictedClass}`;
  document.querySelector("#result-ripeness").textContent = ripenessLabel(data.ripeness);
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
      .map(([name, value]) => `<div class="probability-row"><span title="${escapeHtml(name)}">${escapeHtml(classLabel(name))}</span><span class="bar-track"><span class="bar-fill" style="display:block;width:${Math.max(0, Math.min(100, Number(value) * 100))}%"></span></span><strong>${percent(value)}</strong></div>`)
      .join("");
  }
  feedback("Kết quả đã được lưu vào lịch sử.", false);
}

export function mount() {
  const input = document.querySelector("#image-file");
  const dropZone = document.querySelector("#drop-zone");
  const inputPanel = document.querySelector(".recognition-input");
  const previewWrap = document.querySelector("#image-preview-wrap");
  const button = document.querySelector("#predict-button");
  previewWrap.addEventListener("animationend", (event) => {
    if (event.animationName === "drop-confirm") previewWrap.classList.remove("is-drop-confirmed");
  });
  input.addEventListener("change", () => {
    setFile(input.files?.[0]);
    input.value = "";
  });
  enableDrop(inputPanel);
  document.querySelector("#clear-image").addEventListener("click", () => {
    selectedFile = null;
    input.value = "";
    delete document.documentElement.dataset.ripeness;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    document.querySelector("#image-preview-wrap").classList.add("is-hidden");
    dropZone.classList.remove("is-hidden");
    button.disabled = true;
    resetResult();
  });
  button.addEventListener("click", async () => {
    if (!selectedFile) return;
    const requestFile = selectedFile;
    button.disabled = true;
    button.innerHTML = '<span class="spinner" aria-hidden="true"></span> Đang phân loại';
    document.querySelector("#result-state").textContent = "Đang xử lý ảnh";
    feedback("");
    try {
      const result = await postImage(requestFile);
      if (selectedFile === requestFile) showPrediction(result);
    } catch (error) {
      if (selectedFile !== requestFile) return;
      document.querySelector("#result-state").textContent = "Không hoàn tất";
      document.querySelector("#result-state").className = "result-state is-error";
      feedback(error.message);
    } finally {
      const selectionChanged = selectedFile !== requestFile;
      button.disabled = selectionChanged ? !selectedFile : false;
      button.innerHTML = selectionChanged ? READY_BUTTON_HTML : REPEAT_BUTTON_HTML;
    }
  });
}
