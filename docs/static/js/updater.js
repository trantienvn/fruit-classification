/**
 * updater.js — Quản lý phiên bản ứng dụng và tự động dọn sạch cache client trình duyệt
 * khi phát hiện bản cập nhật mới trên server/Vercel.
 */

const VERSION_ENDPOINT = "./version.json";
const STORAGE_KEY_VERSION = "fruit_app_version";
const STORAGE_KEY_BUILD_TIME = "fruit_app_build_time";
const DEFAULT_VERSION = "1.3.0";

/**
 * Hiển thị thông báo Toast ở góc màn hình.
 */
export function showToast(message, type = "info", duration = 4000) {
  const region = document.querySelector("#toast-region");
  if (!region) {
    console.log(`[Toast ${type}]: ${message}`);
    return;
  }

  const toast = document.createElement("div");
  toast.className = `toast toast-${type} is-${type}`;
  toast.setAttribute("role", "alert");
  toast.textContent = message;

  region.appendChild(toast);

  setTimeout(() => {
    toast.style.transition = "opacity 0.3s ease, transform 0.3s ease";
    toast.style.opacity = "0";
    toast.style.transform = "translateY(8px)";
    setTimeout(() => toast.remove(), 320);
  }, duration);
}

/**
 * Lấy phiên bản cục bộ đang chạy.
 */
export function getCurrentVersion() {
  try {
    return localStorage.getItem(STORAGE_KEY_VERSION) || DEFAULT_VERSION;
  } catch (e) {
    return DEFAULT_VERSION;
  }
}

/**
 * Dọn sạch toàn bộ cache của trình duyệt cho trang web:
 *  - CacheStorage (Service Worker / PWA cache)
 *  - Unregister Service Worker cũ
 *  - sessionStorage
 *  - Cập nhật phiên bản mới vào localStorage (bảo toàn lịch sử dự đoán IndexedDB)
 */
export async function clearClientCache(newVersion = null, newBuildTime = null) {
  console.log("[AutoUpdater] Đang tiến hành dọn sạch cache trình duyệt...");

  // 1. Xóa toàn bộ CacheStorage API
  if (typeof window !== "undefined" && "caches" in window) {
    try {
      const cacheNames = await window.caches.keys();
      await Promise.all(
        cacheNames.map((cacheName) => {
          console.log(`[AutoUpdater] Xóa CacheStorage: ${cacheName}`);
          return window.caches.delete(cacheName);
        })
      );
    } catch (err) {
      console.warn("[AutoUpdater] Không thể xóa CacheStorage:", err);
    }
  }

  // 2. Hủy đăng ký Service Worker cũ (nếu có)
  if (typeof navigator !== "undefined" && "serviceWorker" in navigator) {
    try {
      const registrations = await navigator.serviceWorker.getRegistrations();
      for (const reg of registrations) {
        console.log("[AutoUpdater] Hủy đăng ký ServiceWorker cũ");
        await reg.unregister();
      }
    } catch (err) {
      console.warn("[AutoUpdater] Không thể hủy đăng ký ServiceWorker:", err);
    }
  }

  // 3. Xóa sessionStorage
  try {
    sessionStorage.clear();
  } catch (e) {}

  // 4. Lưu phiên bản mới vào localStorage
  if (newVersion) {
    try {
      localStorage.setItem(STORAGE_KEY_VERSION, newVersion);
      if (newBuildTime) {
        localStorage.setItem(STORAGE_KEY_BUILD_TIME, String(newBuildTime));
      }
    } catch (e) {}
  }
}

/**
 * Kiểm tra xem có phiên bản cập nhật mới từ server không.
 * @param {Object} options
 * @param {boolean} options.autoReload - Tự động tải lại trang nếu phát hiện bản mới
 * @param {boolean} options.notifyIfLatest - Hiển thị toast nếu đang ở bản mới nhất
 */
export async function checkAppUpdate({ autoReload = true, notifyIfLatest = false } = {}) {
  try {
    // Luôn gửi timestamp ngẫu nhiên và headers no-cache để vượt qua HTTP cache
    const url = `${VERSION_ENDPOINT}?_t=${Date.now()}`;
    const res = await fetch(url, {
      method: "GET",
      cache: "no-store",
      headers: {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        Pragma: "no-cache",
      },
    });

    if (!res.ok) return;

    const data = await res.json();
    const serverVersion = data.version;
    const serverBuildTime = data.buildTime || 0;
    const localVersion = localStorage.getItem(STORAGE_KEY_VERSION);
    const localBuildTime = Number(localStorage.getItem(STORAGE_KEY_BUILD_TIME) || 0);

    const isNewVersion =
      !localVersion ||
      localVersion !== serverVersion ||
      (serverBuildTime > 0 && serverBuildTime > localBuildTime);

    if (isNewVersion) {
      console.log(
        `[AutoUpdater] Phát hiện bản cập nhật mới! Server: v${serverVersion} (Build: ${serverBuildTime}) vs Client: v${localVersion || "chưa rõ"}`
      );

      showToast(
        `Đã có phiên bản mới (v${serverVersion}). Đang tự động dọn cache & cập nhật...`,
        "info",
        5000
      );

      // Dọn sạch cache và cập nhật version
      await clearClientCache(serverVersion, serverBuildTime);

      if (autoReload) {
        setTimeout(() => {
          // Bắt buộc tải lại từ server
          window.location.reload();
        }, 1200);
      }
      return true;
    } else {
      if (notifyIfLatest) {
        showToast(`Ứng dụng đang ở phiên bản mới nhất (v${serverVersion}).`, "success", 3000);
      }
      return false;
    }
  } catch (err) {
    console.warn("[AutoUpdater] Không thể kiểm tra cập nhật:", err);
    return false;
  }
}

/**
 * Khởi động tiến trình tự động kiểm tra cập nhật:
 *  - Khi người dùng tải trang
 *  - Khi người dùng chuyển tab quay lại (visibilitychange)
 *  - Kiểm tra định kỳ mỗi 10 phút
 */
export function initAutoUpdater() {
  // 1. Kiểm tra ngay khi khởi động
  setTimeout(() => {
    checkAppUpdate({ autoReload: true, notifyIfLatest: false });
  }, 1000);

  // 2. Kiểm tra khi người dùng mở lại tab
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      checkAppUpdate({ autoReload: true, notifyIfLatest: false });
    }
  });

  // 3. Định kỳ kiểm tra mỗi 10 phút
  setInterval(() => {
    checkAppUpdate({ autoReload: true, notifyIfLatest: false });
  }, 10 * 60 * 1000);
}
