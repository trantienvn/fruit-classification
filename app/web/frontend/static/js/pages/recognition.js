import { postImage } from "../api.js";
import { classLabel, escapeHtml, fruitLabel, percent, ripenessLabel } from "../utils.js";

const MAX_BYTES = 10 * 1024 * 1024;
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

function setFile(file) {
  if (!file) return;
  if (file.size > MAX_BYTES) {
    selectedFile = null;
    document.querySelector("#predict-button").disabled = true;
    document.querySelector("#image-preview-wrap").classList.add("is-hidden");
    document.querySelector("#drop-zone").classList.remove("is-hidden");
    document.querySelector("#result-state").textContent = "Tệp quá lớn";
    feedback("Ảnh vượt quá 10 MB. Hãy chọn một tệp nhỏ hơn.");
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  const preview = document.querySelector("#image-preview");
  preview.src = previewUrl;
  preview.onload = () => {
    document.querySelector("#preview-name").textContent = file.name;
    document.querySelector("#preview-size").textContent = `${(file.size / 1024).toFixed(1)} KB · ${preview.naturalWidth} × ${preview.naturalHeight} px`;
  };
  document.querySelector("#drop-zone").classList.add("is-hidden");
  document.querySelector("#image-preview-wrap").classList.remove("is-hidden");
  document.querySelector("#predict-button").disabled = false;
  resetResult();
}

function showPrediction(data) {
  document.querySelector("#result-empty").classList.add("is-hidden");
  document.querySelector("#result-content").classList.remove("is-hidden");
  document.querySelector("#result-state").textContent = "Đã phân loại";
  document.querySelector("#result-state").className = "result-state is-success";
  document.querySelector("#result-fruit").textContent = fruitLabel(data.fruit_type);
  document.querySelector("#result-class").textContent = data.class_name;
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
  const button = document.querySelector("#predict-button");
  input.addEventListener("change", () => setFile(input.files?.[0]));
  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("is-dragging"));
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
    setFile(event.dataTransfer?.files?.[0]);
  });
  document.querySelector("#clear-image").addEventListener("click", () => {
    selectedFile = null;
    input.value = "";
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    document.querySelector("#image-preview-wrap").classList.add("is-hidden");
    dropZone.classList.remove("is-hidden");
    button.disabled = true;
    resetResult();
  });
  button.addEventListener("click", async () => {
    if (!selectedFile) return;
    button.disabled = true;
    button.innerHTML = '<span class="spinner" aria-hidden="true"></span> Đang phân loại';
    document.querySelector("#result-state").textContent = "Đang xử lý ảnh";
    feedback("");
    try {
      const result = await postImage(selectedFile);
      showPrediction(result);
    } catch (error) {
      document.querySelector("#result-state").textContent = "Không hoàn tất";
      document.querySelector("#result-state").className = "result-state is-error";
      feedback(error.message);
    } finally {
      button.disabled = false;
      button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.8" cy="10.8" r="6.8"/><path d="m16 16 4.5 4.5M10.8 7.5v6.6m-3.3-3.3h6.6"/></svg> Phân loại lại ảnh';
    }
  });
}
