/**
 * predict.worker.js — Web Worker thực thi suy luận ONNX Runtime Web trong luồng riêng biệt,
 * đảm bảo giao diện người dùng hoàn toàn mượt mà, không bị lag/đơ khi xử lý mạng nơ-ron.
 */

/* eslint-disable no-undef */
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js");

// Cấu hình đường dẫn nạp WebAssembly từ CDN jsDelivr
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
// Đặt numThreads = 1 để tương thích 100% với môi trường tĩnh không có SharedArrayBuffer / COOP / COEP
ort.env.wasm.numThreads = 1;

let sessionPromise = null;
let session = null;

async function initSession() {
  if (session) return session;
  if (!sessionPromise) {
    self.postMessage({
      type: "status",
      status: "loading_model",
      message: "Đang nạp mô hình ONNX (~4.2 MB) vào bộ nhớ...",
    });

    sessionPromise = (async () => {
      try {
        const sess = await ort.InferenceSession.create("/models/model.onnx", {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
        session = sess;
        self.postMessage({
          type: "status",
          status: "ready",
          message: "Mô hình ONNX đã nạp xong và sẵn sàng suy luận.",
        });
        return sess;
      } catch (err) {
        self.postMessage({
          type: "status",
          status: "error",
          message: `Lỗi nạp mô hình ONNX: ${err.message}`,
        });
        sessionPromise = null;
        throw err;
      }
    })();
  }
  return sessionPromise;
}

// Khởi chạy nạp trước mô hình ở chế độ ngầm
initSession().catch(() => {});

// Xử lý thông điệp từ main thread
self.onmessage = async (event) => {
  const { type, id, tensorData } = event.data;

  if (type === "init") {
    try {
      await initSession();
      self.postMessage({ type: "init_success", id });
    } catch (err) {
      self.postMessage({ type: "init_error", id, error: err.message });
    }
    return;
  }

  if (type === "predict") {
    try {
      const sess = await initSession();
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
