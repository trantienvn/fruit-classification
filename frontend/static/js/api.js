/**
 * api.js — Tầng tương thích Client-Side (Client-Only Facade).
 * Chuyển toàn bộ các lời gọi API cũ (/api/*) sang thực thi trực tiếp trên trình duyệt
 * qua onnxruntime-web (predict.js) và IndexedDB (history-store.js).
 */

import { predict, onModelStatus } from "./predict.js";
import {
  addHistory,
  listHistory,
  clearHistory,
  getStatistics,
} from "./history-store.js";
import { FULL_CLASS_MAP_VI } from "./labels.js";

export class ApiError extends Error {
  constructor(message, status = 400, code = "client_error") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

/**
 * Giả lập endpoint GET cho các dịch vụ frontend không còn backend.
 */
export async function get(path, params = {}) {
  // 1. Trạng thái Health check
  if (path.includes("/api/health") || path === "/health") {
    return {
      status: "ready",
      model_ready: true,
      history_ready: true,
      model_name: "MobileNetV3-Small + PyTorch MLP (ONNX Wasm)",
      backbone: "mobilenetv3_small",
      device: "WebAssembly (Trình duyệt)",
      n_classes: 9,
    };
  }

  // 2. Metadata mô hình từ /models/config.json
  if (path.includes("/api/model-info") || path === "/model-info") {
    let config = {};
    try {
      const res = await fetch("/models/config.json");
      if (res.ok) {
        config = await res.json();
      }
    } catch (e) {
      console.warn("Không thể đọc /models/config.json:", e);
    }

    const classNames = config.class_names || Object.keys(FULL_CLASS_MAP_VI);

    return {
      model_type: "deep_learning_onnx",
      model_name: "CNN MobileNetV3-Small + PyTorch MLP (ONNX WebAssembly)",
      backbone: config.backbone || "mobilenetv3_small",
      embedding_dim: config.embedding_dim || 576,
      input_size: config.crop_size || [224, 224],
      test_accuracy: config.test_accuracy || 0.988889,
      test_f1: config.test_f1 || 0.988746,
      total_epochs: config.total_epochs || 15,
      device: "WebAssembly / Browser Client",
      class_names: classNames,
      n_classes: classNames.length,
      use_pca: config.use_pca ?? true,
      pca_ratio: config.pca_ratio ?? 0.95,
      reports: [
        {
          title: "Sơ đồ xu hướng Loss theo Epoch",
          url: "/static/img/loss_curve.png",
          type: "loss_curve",
          description: "Đường cong Cross-Entropy Loss qua các Epoch huấn luyện.",
        },
        {
          title: "Tiến trình huấn luyện & Tốc độ hội tụ",
          url: "/static/img/epoch_training.png",
          type: "training_metrics",
          description: "Mối tương quan giữa hàm mất mát và độ chính xác phân loại.",
        },
      ],
    };
  }

  // 3. Lịch sử dự đoán
  if (path.includes("/api/history") || path === "/history") {
    return listHistory(params);
  }

  // 4. Thống kê tổng hợp
  if (path.includes("/api/statistics") || path === "/statistics") {
    return getStatistics();
  }

  // Fallback đọc file tĩnh thông thường
  const res = await fetch(path);
  if (!res.ok) throw new ApiError(`Không tìm thấy tài nguyên: ${path}`, res.status);
  return res.json();
}

/**
 * Thay thế upload ảnh POST /api/predict bằng suy luận trực tiếp ONNX trong trình duyệt.
 */
export async function postImage(file) {
  const result = await predict(file);
  // Lưu vào IndexedDB lịch sử
  await addHistory(result, file);
  return result;
}

/**
 * Xóa kho lịch sử IndexedDB.
 */
export async function removeHistory() {
  return clearHistory();
}

export { onModelStatus };
