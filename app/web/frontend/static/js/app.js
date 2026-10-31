import { get } from "./api.js";

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
    pill.querySelector("span").textContent = modelName ? `API hoạt động · ${modelName}` : "API hoạt động";
    dot.className = "state-indicator is-ready";
    text.textContent = "API và model sẵn sàng";
  } else if (status === "degraded") {
    pill.className = "backend-pill is-error";
    pill.querySelector("span").textContent = "Dịch vụ chưa sẵn sàng";
    dot.className = "state-indicator is-error";
    text.textContent = "Kiểm tra model và lịch sử";
  } else {
    pill.className = "backend-pill is-error";
    pill.querySelector("span").textContent = "Không kết nối được API";
    dot.className = "state-indicator is-error";
    text.textContent = "API không phản hồi";
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
    const [markup, controller] = await Promise.all([
      fetch(`/pages/${routeName}.html`).then((response) => {
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
  menuToggle.addEventListener("click", () => {
    const open = sidebar.classList.toggle("is-open");
    menuToggle.setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("click", (event) => {
    if (sidebar.classList.contains("is-open") && !sidebar.contains(event.target) && !menuToggle.contains(event.target)) {
      sidebar.classList.remove("is-open");
      menuToggle.setAttribute("aria-expanded", "false");
    }
  });
  refreshStatus();
  window.setInterval(refreshStatus, 30000);
  renderRoute();
});
