/**
 * history-store.js — Quản lý kho lịch sử phân loại trực tiếp trong trình duyệt qua IndexedDB
 * (kèm fallback localStorage an toàn), thay thế hoàn toàn SQLite backend khi chạy tĩnh trên Vercel.
 */

import { CLASS_NAMES, FRUIT_NAMES, RIPENESS_NAMES } from "./labels.js";

const DB_NAME = "fruit_classification_db";
const STORE_NAME = "predictions";
const DB_VERSION = 1;
const MAX_HISTORY_ITEMS = 50;

/**
 * Tạo thumbnail nhỏ (tối đa 70x70 px) dạng DataURL nén JPEG để tiết kiệm dung lượng lưu trữ.
 */
export async function createThumbnail(fileOrBlob) {
  try {
    const bitmap = await createImageBitmap(fileOrBlob);
    const maxSize = 70;
    let w = bitmap.width;
    let h = bitmap.height;

    if (w > h) {
      if (w > maxSize) {
        h = Math.round((h * maxSize) / w);
        w = maxSize;
      }
    } else {
      if (h > maxSize) {
        w = Math.round((w * maxSize) / h);
        h = maxSize;
      }
    }

    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0, w, h);
    return canvas.toDataURL("image/jpeg", 0.75);
  } catch (err) {
    return null;
  }
}

/**
 * Mở kết nối IndexedDB.
 */
function openDB() {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("IndexedDB không khả dụng"));
      return;
    }

    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "id" });
        store.createIndex("created_at", "created_at", { unique: false });
        store.createIndex("class_name", "class_name", { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Thêm một mục lịch sử dự đoán mới.
 */
export async function addHistory(predictionData, fileOrBlob = null) {
  const item = {
    ...predictionData,
    id: predictionData.id || Date.now(),
    thumbnail: fileOrBlob ? await createThumbnail(fileOrBlob) : null,
  };

  try {
    const db = await openDB();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      store.put(item);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });

    // Cắt tỉa nếu vượt quá 50 mục
    await pruneHistory();
  } catch (err) {
    // Fallback sang localStorage nếu IndexedDB lỗi
    try {
      const existing = getLocalStorageItems();
      existing.unshift(item);
      if (existing.length > MAX_HISTORY_ITEMS) existing.length = MAX_HISTORY_ITEMS;
      localStorage.setItem("fruit_history", JSON.stringify(existing));
    } catch (e) {
      console.warn("Không thể lưu lịch sử vào localStorage:", e);
    }
  }

  return item;
}

/**
 * Cắt bớt các mục cũ nếu vượt quá 50 mục.
 */
async function pruneHistory() {
  try {
    const db = await openDB();
    const items = await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });

    if (items.length > MAX_HISTORY_ITEMS) {
      items.sort((a, b) => b.id - a.id);
      const toDelete = items.slice(MAX_HISTORY_ITEMS);
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      toDelete.forEach((it) => store.delete(it.id));
    }
  } catch (e) {
    // Bỏ qua lỗi prune
  }
}

function getLocalStorageItems() {
  try {
    const raw = localStorage.getItem("fruit_history");
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    return [];
  }
}

/**
 * Liệt kê lịch sử có hỗ trợ tìm kiếm, lọc và phân trang.
 */
export async function listHistory({
  limit = 50,
  offset = 0,
  query = "",
  className = "",
} = {}) {
  let allItems = [];

  try {
    const db = await openDB();
    allItems = await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    allItems = getLocalStorageItems();
  }

  // Sắp xếp mới nhất lên đầu
  allItems.sort((a, b) => (b.id || 0) - (a.id || 0));

  let filtered = allItems;
  if (query) {
    const q = query.toLowerCase().trim();
    filtered = filtered.filter((it) =>
      (it.filename && it.filename.toLowerCase().includes(q)) ||
      (it.class_name && it.class_name.toLowerCase().includes(q)) ||
      (it.class_name_vi && it.class_name_vi.toLowerCase().includes(q))
    );
  }
  if (className) {
    filtered = filtered.filter((it) => it.class_name === className);
  }

  const total = filtered.length;
  const items = filtered.slice(offset, offset + limit);

  return { items, total, limit, offset };
}

/**
 * Xóa toàn bộ lịch sử.
 */
export async function clearHistory() {
  let deletedCount = 0;
  try {
    const db = await openDB();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const countReq = store.count();
      countReq.onsuccess = () => {
        deletedCount = countReq.result;
        store.clear();
      };
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch (err) {
    const items = getLocalStorageItems();
    deletedCount = items.length;
    localStorage.removeItem("fruit_history");
  }
  return { deleted: deletedCount };
}

/**
 * Tính toán thống kê phục vụ trang Tổng quan (Dashboard) và Thống kê (Statistics).
 */
export async function getStatistics() {
  const { items } = await listHistory({ limit: 500 });

  const total = items.length;
  const uniqueClassSet = new Set(items.map((i) => i.class_name).filter(Boolean));

  const byClassMap = {};
  CLASS_NAMES.forEach((cls) => {
    byClassMap[cls] = 0;
  });

  const byFruitMap = {};
  FRUIT_NAMES.forEach((fr) => {
    byFruitMap[fr] = 0;
  });

  const byRipenessMap = {};
  RIPENESS_NAMES.forEach((rp) => {
    byRipenessMap[rp] = 0;
  });

  const byDayMap = {};

  items.forEach((item) => {
    const cls = item.class_name;
    if (byClassMap[cls] !== undefined) byClassMap[cls]++;

    const fr = item.fruit_type;
    if (byFruitMap[fr] !== undefined) byFruitMap[fr]++;

    const rp = item.ripeness;
    if (byRipenessMap[rp] !== undefined) byRipenessMap[rp]++;

    if (item.created_at) {
      const day = item.created_at.slice(0, 10);
      byDayMap[day] = (byDayMap[day] || 0) + 1;
    }
  });

  const by_class = Object.entries(byClassMap).map(([name, count]) => ({
    name,
    class_name: name,
    count,
  }));
  const by_fruit = Object.entries(byFruitMap).map(([name, count]) => ({
    name,
    fruit: name,
    count,
  }));
  const by_ripeness = Object.entries(byRipenessMap).map(([name, count]) => ({
    name,
    ripeness: name,
    count,
  }));
  const by_day = Object.entries(byDayMap)
    .sort((a, b) => a[0].localeCompare(b[0]))
    .slice(-7)
    .map(([date, count]) => ({ date, count }));

  return {
    total_predictions: total,
    unique_classes: uniqueClassSet.size,
    latest_at: items[0]?.created_at || null,
    by_class,
    by_fruit,
    by_ripeness,
    by_day,
  };
}
