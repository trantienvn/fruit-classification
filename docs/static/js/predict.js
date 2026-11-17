/**
 * predict.js — Module suy luận Deep Learning phía client (In-Browser Inference).
 * Tiền xử lý ảnh qua Canvas chuẩn hóa ImageNet và giao tiếp với Web Worker chạy onnxruntime-web.
 */

import { CLASS_NAMES, formatPredictionData } from "./labels.js";

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const TARGET_WIDTH = 224;
const TARGET_HEIGHT = 224;

let worker = null;
let currentRequestId = 0;
const pendingRequests = new Map();
const statusListeners = new Set();
let workerStatus = "idle"; // "idle" | "loading_model" | "ready" | "error"
let activeModelName = "ResNet-18 (Mặc định)";

/**
 * Đăng ký lắng nghe trạng thái nạp mô hình của Worker.
 */
export function onModelStatus(callback) {
  statusListeners.add(callback);
  callback({ status: workerStatus, model: activeModelName });
  return () => statusListeners.delete(callback);
}

function notifyStatus(status, message, modelFile) {
  workerStatus = status;
  for (const listener of statusListeners) {
    try {
      listener({ status, message, modelFile });
    } catch (e) {
      console.error(e);
    }
  }
}

/**
 * Khởi tạo Web Worker duy nhất cho toàn bộ phiên làm việc.
 */
function getWorker() {
  if (worker) return worker;

  try {
    worker = new Worker(new URL("./predict.worker.js", import.meta.url));

    worker.onmessage = (event) => {
      const { type, id, status, message, modelFile, probs, latencyMs, error } = event.data;

      if (type === "status") {
        notifyStatus(status, message, modelFile);
        return;
      }

      if (type === "switch_model_success") {
        if (pendingRequests.has(id)) {
          const { resolve } = pendingRequests.get(id);
          pendingRequests.delete(id);
          resolve({ modelFile });
        }
        return;
      }

      if (type === "switch_model_error") {
        if (pendingRequests.has(id)) {
          const { reject } = pendingRequests.get(id);
          pendingRequests.delete(id);
          reject(new Error(error || "Lỗi chuyển đổi mô hình."));
        }
        return;
      }

      if (!pendingRequests.has(id)) return;
      const { resolve, reject } = pendingRequests.get(id);
      pendingRequests.delete(id);

      if (type === "predict_success") {
        resolve({ probs, latencyMs, modelFile });
      } else {
        reject(new Error(error || "Lỗi suy luận ONNX."));
      }
    };

    worker.onerror = (err) => {
      notifyStatus("error", `Lỗi Worker: ${err.message}`);
    };

    // Yêu cầu nạp trước mô hình
    worker.postMessage({ type: "init" });
  } catch (err) {
    notifyStatus("error", `Không thể khởi tạo Web Worker: ${err.message}`);
    throw err;
  }

  return worker;
}

/**
 * Chuyển đổi mô hình ONNX đang chạy (ResNet-18, MobileNetV3-Small, MobileNetV2).
 */
export async function switchModel(modelFileName) {
  const activeWorker = getWorker();
  const requestId = ++currentRequestId;

  return new Promise((resolve, reject) => {
    pendingRequests.set(requestId, { resolve, reject });
    activeWorker.postMessage({
      type: "switch_model",
      id: requestId,
      modelFile: modelFileName,
    });
  });
}

/**
 * Tiền xử lý ảnh đầu vào bằng Canvas:
 *  - Tự động xoay ảnh theo EXIF orientation.
 *  - Vẽ trên nền trắng 224x224 (xử lý tốt ảnh trong suốt PNG).
 *  - Chuyển RGB sang chuẩn float32 CHW: (3, 224, 224).
 *  - Chuẩn hóa ImageNet Mean [0.485, 0.456, 0.406] / Std [0.229, 0.224, 0.225].
 */
export async function preprocessImage(fileOrBlob) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(fileOrBlob, { imageOrientation: "from-image" });
  } catch (e) {
    bitmap = await createImageBitmap(fileOrBlob);
  }

  let canvas;
  if (typeof OffscreenCanvas !== "undefined") {
    canvas = new OffscreenCanvas(TARGET_WIDTH, TARGET_HEIGHT);
  } else {
    canvas = document.createElement("canvas");
    canvas.width = TARGET_WIDTH;
    canvas.height = TARGET_HEIGHT;
  }

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
  ctx.drawImage(bitmap, 0, 0, TARGET_WIDTH, TARGET_HEIGHT);

  const imgData = ctx.getImageData(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
  const data = imgData.data;

  const numPixels = TARGET_WIDTH * TARGET_HEIGHT;
  const float32Tensor = new Float32Array(3 * numPixels);

  const rOffset = 0;
  const gOffset = numPixels;
  const bOffset = numPixels * 2;

  for (let i = 0; i < numPixels; i++) {
    const src = i * 4;
    const r = data[src] / 255.0;
    const g = data[src + 1] / 255.0;
    const b = data[src + 2] / 255.0;

    float32Tensor[rOffset + i] = (r - MEAN[0]) / STD[0];
    float32Tensor[gOffset + i] = (g - MEAN[1]) / STD[1];
    float32Tensor[bOffset + i] = (b - MEAN[2]) / STD[2];
  }

  return float32Tensor;
}

/**
 * Thực thi suy luận ảnh trên trình duyệt.
 * @param {File|Blob} fileOrBlob - Tệp ảnh đầu vào
 * @returns {Promise<Object>} Dữ liệu kết quả phân loại chuẩn hóa
 */
export async function predict(fileOrBlob) {
  const t0 = performance.now();
  const tensorData = await preprocessImage(fileOrBlob);

  const activeWorker = getWorker();
  const requestId = ++currentRequestId;

  const { probs, latencyMs, modelFile } = await new Promise((resolve, reject) => {
    pendingRequests.set(requestId, { resolve, reject });
    activeWorker.postMessage(
      {
        type: "predict",
        id: requestId,
        tensorData,
      },
      [tensorData.buffer]
    );
  });

  const totalTimeMs = Math.round(performance.now() - t0);
  const filename = fileOrBlob && fileOrBlob.name ? fileOrBlob.name : "anh-trai-cay.jpg";

  // Định dạng kết quả từ mảng xác suất
  const formatted = formatPredictionData(probs, CLASS_NAMES, latencyMs, filename);
  formatted.latency_ms = Math.round(latencyMs);
  formatted.total_client_ms = totalTimeMs;
  formatted.model_file = modelFile || "model.onnx";

  return formatted;
}
