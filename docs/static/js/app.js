import { get } from "./api.js";
import { initAutoUpdater, getCurrentVersion, checkAppUpdate, clearClientCache, showToast } from "./updater.js";

const routes = {
  dashboard: { title: "Tổng quan", module: () => import("./pages/dashboard.js") },
  recognition: { title: "Nhận diện ảnh", module: () => import("./pages/recognition.js") },
  history: { title: "Lịch sử", module: () => import("./pages/history.js") },
  statistics: { title: "Thống kê", module: () => import("./pages/statistics.js") },
  "model-info": { title: "Thông tin mô hình", module: () => import("./pages/model-info.js") },
};

const view = document.querySelector("#page-view");
const sidebar = document.querySelector("#sidebar");
const menuToggle = document.querySelector("#menu-toggle");
let renderToken = 0;

function setStatus(status, modelName) {
  const pill = document.querySelector("#backend-pill");
  const dot = document.querySelector("#sidebar-state-dot");
  const text = document.querySelector("#sidebar-state-text");
  if (status === "ready") {
    pill.className = "backend-pill is-ready";
    pill.querySelector("span").textContent = modelName ? `ONNX Wasm · ${modelName}` : "ONNX Wasm sẵn sàng";
    dot.className = "state-indicator is-ready";
    text.textContent = "Mô hình ONNX sẵn sàng";
  } else if (status === "degraded" || status === "loading") {
    pill.className = "backend-pill";
    pill.querySelector("span").textContent = "Đang nạp mô hình...";
    dot.className = "state-indicator";
    text.textContent = "Đang tải ONNX Wasm";
  } else {
    pill.className = "backend-pill is-error";
    pill.querySelector("span").textContent = "Mô hình chưa sẵn sàng";
    dot.className = "state-indicator is-error";
    text.textContent = "Kiểm tra tệp ONNX";
  }
}

async function refreshStatus() {
  try {
    const health = await get("/api/health");
    setStatus(health.status, health.model_name);
  } catch {
    setStatus("offline");
  }
}

async function renderRoute() {
  let routeName = window.location.hash.slice(1).split("?")[0] || "dashboard";
  if (!routes[routeName]) {
    window.history.replaceState(null, "", "#/dashboard");
    routeName = "dashboard";
  }
  const token = ++renderToken;
  const route = routes[routeName];
  document.querySelector("#page-title").textContent = route.title;
  document.querySelectorAll("[data-route]").forEach((link) => {
    const active = link.dataset.route === routeName;
    link.classList.toggle("is-active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  sidebar.classList.remove("is-open");
  menuToggle.setAttribute("aria-expanded", "false");
  view.innerHTML = '<div class="page-loading"><span class="spinner"></span><span>Đang mở trang</span></div>';

  try {
    const currentVer = getCurrentVersion();
    const [markup, controller] = await Promise.all([
      fetch(`./pages/${routeName}.html?v=${encodeURIComponent(currentVer)}`, { cache: "no-cache" }).then((response) => {
        if (!response.ok) throw new Error("Không tải được nội dung trang.");
        return response.text();
      }),
      route.module(),
    ]);
    if (token !== renderToken) return;
    view.innerHTML = markup;
    await controller.mount(view, { refreshStatus });
    view.focus({ preventScroll: true });
  } catch (error) {
    if (token !== renderToken) return;
    view.innerHTML = '<div class="panel page-error"><strong>Không mở được trang</strong><p></p><button class="button button-primary" type="button">Tải lại</button></div>';
    view.querySelector("p").textContent = String(error.message || error);
    view.querySelector("button").addEventListener("click", () => window.location.reload());
  }
}

window.addEventListener("hashchange", renderRoute);
window.addEventListener("DOMContentLoaded", () => {
  const appShell = document.querySelector(".app-shell");
  const collapseBtn = document.querySelector("#nav-collapse-btn");

  // Thiết lập trạng thái thu gọn điều hướng
  function setNavCollapsed(collapsed) {
    if (!appShell) return;
    if (collapsed) {
      appShell.classList.add("is-nav-collapsed");
      if (collapseBtn) {
        collapseBtn.setAttribute("title", "Mở rộng thanh điều hướng (Ctrl+B)");
        collapseBtn.setAttribute("aria-label", "Mở rộng thanh điều hướng");
      }
      try {
        localStorage.setItem("fruit_nav_collapsed", "true");
      } catch (e) {}
    } else {
      appShell.classList.remove("is-nav-collapsed");
      if (collapseBtn) {
        collapseBtn.setAttribute("title", "Thu gọn thanh điều hướng (Ctrl+B)");
        collapseBtn.setAttribute("aria-label", "Thu gọn thanh điều hướng");
      }
      try {
        localStorage.setItem("fruit_nav_collapsed", "false");
      } catch (e) {}
    }
  }

  function toggleNavCollapsed() {
    if (!appShell) return;
    const isCurrentlyCollapsed = appShell.classList.contains("is-nav-collapsed");
    setNavCollapsed(!isCurrentlyCollapsed);
  }

  // Khôi phục trạng thái thu gọn đã lưu từ localStorage trên màn hình Desktop
  try {
    const savedCollapsed = localStorage.getItem("fruit_nav_collapsed") === "true";
    if (savedCollapsed && window.innerWidth > 800) {
      setNavCollapsed(true);
    }
  } catch (e) {}

  // Bắt sự kiện bấm nút thu gọn trong sidebar
  if (collapseBtn) {
    collapseBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleNavCollapsed();
    });
  }

  // Bắt sự kiện nút menu toggle trên topbar:
  // - Trên Desktop (> 800px): Thu gọn / Mở rộng sidebar
  // - Trên Mobile/Tablet (<= 800px): Mở / Đóng thanh drawer
  if (menuToggle) {
    menuToggle.addEventListener("click", (e) => {
      e.stopPropagation();
      if (window.innerWidth > 800) {
        toggleNavCollapsed();
      } else {
        const open = sidebar.classList.toggle("is-open");
        menuToggle.setAttribute("aria-expanded", String(open));
      }
    });
  }

  // Phím tắt Ctrl + B (hoặc Cmd + B) để thu gọn / mở rộng nhanh
  window.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
      e.preventDefault();
      toggleNavCollapsed();
    }
  });

  // Tự động đóng drawer khi click ra ngoài trên màn hình nhỏ
  document.addEventListener("click", (event) => {
    if (
      window.innerWidth <= 800 &&
      sidebar.classList.contains("is-open") &&
      !sidebar.contains(event.target) &&
      !menuToggle.contains(event.target)
    ) {
      sidebar.classList.remove("is-open");
      menuToggle.setAttribute("aria-expanded", "false");
    }
  });

  refreshStatus();
  window.setInterval(refreshStatus, 30000);
  renderRoute();

  // Khởi động tiến trình tự động phát hiện cập nhật mới và xóa cache trình duyệt
  initAutoUpdater();

  // Expose API quản lý phiên bản cho console và UI
  window.fruitApp = {
    checkAppUpdate: () => checkAppUpdate({ autoReload: true, notifyIfLatest: true }),
    clearClientCache: async () => {
      await clearClientCache();
      showToast("Đã dọn sạch cache trình duyệt thành công!", "success");
      setTimeout(() => window.location.reload(), 1000);
    },
    getCurrentVersion,
  };
});
