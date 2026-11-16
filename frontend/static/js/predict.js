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

/**
 * Đăng ký lắng nghe trạng thái nạp mô hình của Worker.
 */
export function onModelStatus(callback) {
  statusListeners.add(callback);
  callback({ status: workerStatus });
  return () => statusListeners.delete(callback);
}

function notifyStatus(status, message) {
  workerStatus = status;
  for (const listener of statusListeners) {
    try {
      listener({ status, message });
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
      const { type, id, status, message, probs, latencyMs, error } = event.data;

      if (type === "status") {
        notifyStatus(status, message);
        return;
      }

      if (!pendingRequests.has(id)) return;
      const { resolve, reject } = pendingRequests.get(id);
      pendingRequests.delete(id);

      if (type === "predict_success") {
        resolve({ probs, latencyMs });
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
  // Nền trắng cho ảnh trong suốt
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
  ctx.drawImage(bitmap, 0, 0, TARGET_WIDTH, TARGET_HEIGHT);

  const imgData = ctx.getImageData(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
  const data = imgData.data; // RGBA uint8 (224*224*4)

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

  const { probs, latencyMs } = await new Promise((resolve, reject) => {
    pendingRequests.set(requestId, { resolve, reject });
    // Chuyển mảng tensorData dạng transferable buffer để tăng tốc độ truyền giữa các luồng
    activeWorker.postMessage(
      {
        type: "predict",
        id: requestId,
        tensorData,
      },
      [tensorData.buffer]
    );
  });

  const totalTimeMs = performance.now() - t0;

  // Lập từ điển xác suất theo 9 nhãn
  const probMap = {};
  CLASS_NAMES.forEach((cls, idx) => {
    probMap[cls] = probs[idx] ?? 0;
  });

  // Tìm lớp có xác suất cao nhất
  let bestIdx = 0;
  let maxP = -1;
  probs.forEach((p, idx) => {
    if (p > maxP) {
      maxP = p;
      bestIdx = idx;
    }
  });

  const predictedClass = CLASS_NAMES[bestIdx] || "cam-chin";

  // Danh sách Top 3
  const top3 = Object.entries(probMap)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([label, p]) => ({ label, p }));

  const formattedData = formatPredictionData({
    className: predictedClass,
    probabilities: probMap,
    latencyMs: totalTimeMs,
    filename: fileOrBlob.name || "anh-tai-len.jpg",
  });

  return {
    ...formattedData,
    probs,
    top3,
    inferenceLatencyMs: latencyMs,
    totalLatencyMs: totalTimeMs,
  };
}

// Tự động khởi tạo worker sẵn sàng
try {
  getWorker();
} catch (e) {
  // Môi trường không hỗ trợ Web Worker (ví dụ SSR/node)
}
