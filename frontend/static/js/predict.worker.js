/**
 * predict.worker.js — Web Worker thực thi suy luận ONNX Runtime Web trong luồng riêng biệt,
 * đảm bảo giao diện người dùng hoàn toàn mượt mà, không bị lag/đơ khi xử lý mạng nơ-ron.
 *
 * Hỗ trợ chuyển đổi linh hoạt giữa 3 kiến trúc:
 *   - model_resnet18.onnx (Mặc định · Acc 98.15%)
 *   - model_mobilenetv3_small.onnx (Siêu nhẹ · 4.2 MB)
 *   - model_mobilenetv2.onnx (Cân bằng · 10.5 MB)
 */

/* eslint-disable no-undef */
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js");

// Cấu hình đường dẫn nạp WebAssembly từ CDN jsDelivr
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
// Đặt numThreads = 1 để tương thích 100% với môi trường tĩnh không có SharedArrayBuffer / COOP / COEP
ort.env.wasm.numThreads = 1;

let sessionPromise = null;
let session = null;
let currentModelFile = "model_resnet18.onnx";

function getModelUrl(fileName) {
  // Tạo URL tương đối chuẩn xác từ vị trí hiện tại
  try {
    return new URL(`../../models/${fileName}`, self.location.href).href;
  } catch (e) {
    return `/models/${fileName}`;
  }
}

const CACHE_NAME = "fruit-models-cache-v1";
const MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000; // 7 ngày

async function getModelBuffer(modelUrl) {
  try {
    const cache = await caches.open(CACHE_NAME);
    const cachedResponse = await cache.match(modelUrl);

    if (cachedResponse) {
      const cachedTime = cachedResponse.headers.get("x-cache-time");
      const now = Date.now();
      if (cachedTime && (now - Number(cachedTime) < MAX_AGE_MS)) {
        return await cachedResponse.arrayBuffer();
      }
    }

    const response = await fetch(modelUrl);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const blob = await response.blob();

    const headers = new Headers(response.headers);
    headers.set("x-cache-time", Date.now().toString());
    const cacheResponse = new Response(blob, {
      status: response.status,
      statusText: response.statusText,
      headers: headers
    });

    await cache.put(modelUrl, cacheResponse.clone());
    return await cacheResponse.arrayBuffer();
  } catch (err) {
    const cache = await caches.open(CACHE_NAME);
    const cachedResponse = await cache.match(modelUrl);
    if (cachedResponse) {
      return await cachedResponse.arrayBuffer();
    }
    throw err;
  }
}

async function initSession(modelFile = currentModelFile) {
  currentModelFile = modelFile;
  const modelUrl = getModelUrl(modelFile);

  self.postMessage({
    type: "status",
    status: "loading_model",
    message: `Đang nạp mô hình ${modelFile} (kiểm tra cache client 7 ngày)...`,
  });

  sessionPromise = (async () => {
    try {
      const modelBuffer = await getModelBuffer(modelUrl);
      const sess = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      session = sess;
      self.postMessage({
        type: "status",
        status: "ready",
        modelFile,
        message: `Mô hình ${modelFile} đã sẵn sàng suy luận.`,
      });
      return sess;
    } catch (err) {
      self.postMessage({
        type: "status",
        status: "error",
        modelFile,
        message: `Lỗi nạp mô hình ${modelFile}: ${err.message}`,
      });
      sessionPromise = null;
      throw err;
    }
  })();

  return sessionPromise;
}

// Khởi chạy nạp trước mô hình ở chế độ ngầm
initSession().catch(() => {});

// Xử lý thông điệp từ main thread
self.onmessage = async (event) => {
  const { type, id, tensorData, modelFile } = event.data;

  if (type === "init") {
    try {
      await initSession();
      self.postMessage({ type: "init_success", id });
    } catch (err) {
      self.postMessage({ type: "init_error", id, error: err.message });
    }
    return;
  }

  if (type === "switch_model") {
    try {
      session = null;
      sessionPromise = null;
      await initSession(modelFile);
      self.postMessage({ type: "switch_model_success", id, modelFile });
    } catch (err) {
      self.postMessage({ type: "switch_model_error", id, error: err.message });
    }
    return;
  }

  if (type === "predict") {
    try {
      const sess = await (session ? Promise.resolve(session) : initSession());
      const inputTensor = new ort.Tensor("float32", tensorData, [1, 3, 224, 224]);

      const t0 = performance.now();
      const outputMap = await sess.run({ input: inputTensor });
      const latencyMs = performance.now() - t0;

      const outputTensor = outputMap.logits;
      const logits = Array.from(outputTensor.data);

      // Softmax ổn định số học (subtract max logit)
      const maxVal = Math.max(...logits);
      const exps = logits.map((v) => Math.exp(v - maxVal));
      const sumExp = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map((v) => v / sumExp);

      self.postMessage({
        type: "predict_success",
        id,
        logits,
        probs,
        latencyMs,
        modelFile: currentModelFile,
      });
    } catch (err) {
      self.postMessage({
        type: "predict_error",
        id,
        error: err.message || String(err),
      });
    }
  }
};
