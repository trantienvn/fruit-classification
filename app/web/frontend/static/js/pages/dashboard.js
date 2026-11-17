import { get } from "../api.js";
import { classLabel, escapeHtml, formatDate, number, percent } from "../utils.js";

function renderDistribution(rows) {
  const container = document.querySelector("#dashboard-distribution");
  const nonzero = (rows || []).filter((row) => row.count > 0).sort((a, b) => b.count - a.count).slice(0, 5);
  if (!nonzero.length) {
    container.innerHTML = '<p class="empty-inline">Chưa có ảnh nào được phân loại. Tải ảnh đầu tiên để bắt đầu.</p>';
    return;
  }
  const max = Math.max(...nonzero.map((row) => row.count));
  container.innerHTML = nonzero.map((row) => `<div class="compact-bar">
    <span class="compact-bar-label" title="${escapeHtml(row.class_name)}">${escapeHtml(classLabel(row.class_name))}</span>
    <span class="bar-track"><span class="bar-fill" style="width:${Math.max(3, row.count / max * 100)}%"></span></span>
    <span class="compact-bar-count">${number(row.count)}</span>
  </div>`).join("");
}

function renderRecent(items) {
  const container = document.querySelector("#dashboard-recent");
  if (!items?.length) {
    container.innerHTML = '<p class="empty-inline">Lịch sử chưa có dữ liệu. Kết quả mới sẽ xuất hiện tại đây.</p>';
    return;
  }
  container.innerHTML = items.slice(0, 4).map((item) => `<div class="recent-row">
    <div><strong>${escapeHtml(classLabel(item.class_name))}</strong><small>${escapeHtml(item.filename)}</small></div>
    <time datetime="${escapeHtml(item.created_at)}">${escapeHtml(formatDate(item.created_at))}</time>
  </div>`).join("");
}

export async function mount(_root, { refreshStatus }) {
  const [statistics, history, modelInfo] = await Promise.allSettled([
    get("/api/statistics"),
    get("/api/history", { limit: 5 }),
    get("/api/model-info"),
  ]);
  if (statistics.status === "fulfilled") {
    const data = statistics.value;
    document.querySelector("#metric-total").textContent = number(data.total_predictions);
    document.querySelector("#metric-classes").textContent = `${data.unique_classes} / 9`;
    renderDistribution(data.by_class);
  } else {
    document.querySelector("#dashboard-distribution").innerHTML = '<p class="empty-inline">Không tải được thống kê. Kiểm tra kết nối API.</p>';
  }
  if (history.status === "fulfilled") renderRecent(history.value.items);
  else document.querySelector("#dashboard-recent").innerHTML = '<p class="empty-inline">Không tải được lịch sử.</p>';

  if (modelInfo.status === "fulfilled") {
    const info = modelInfo.value;
    const accuracy = info.test_metrics?.accuracy;
    document.querySelector("#metric-accuracy").textContent = percent(accuracy);
    document.querySelector("#dashboard-model-name").textContent = `${info.model_name || "Model"} đang hoạt động`;
    document.querySelector("#dashboard-model-detail").textContent = `${number(info.n_features)} đặc trưng · ${number(info.n_classes)} lớp · PCA ${info.use_pca ? "đang bật" : "không dùng"}`;
  } else {
    document.querySelector("#metric-accuracy").textContent = "—";
    document.querySelector("#dashboard-model-name").textContent = "Chưa có model sẵn sàng";
    document.querySelector("#dashboard-model-detail").textContent = "Tạo artifact bằng lệnh huấn luyện trong tài liệu dự án.";
  }
  await refreshStatus();
}
