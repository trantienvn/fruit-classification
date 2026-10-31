import { get, removeHistory } from "../api.js";
import { classLabel, escapeHtml, formatDate, fruitLabel, number, percent, ripenessLabel } from "../utils.js";

const PAGE_SIZE = 50;
let offset = 0;
let searchTimer;

async function loadHistory() {
  const query = document.querySelector("#history-search").value.trim();
  const className = document.querySelector("#history-class").value;
  const body = document.querySelector("#history-rows");
  body.innerHTML = '<tr><td colspan="6" class="loading-cell">Đang lấy lịch sử…</td></tr>';
  try {
    const data = await get("/api/history", { q: query, class_name: className, limit: PAGE_SIZE, offset });
    document.querySelector("#history-count").textContent = `${number(data.total)} lượt`;
    document.querySelector("#clear-history").disabled = data.total === 0;
    const page = Math.floor(offset / PAGE_SIZE) + 1;
    const pages = Math.max(1, Math.ceil(data.total / PAGE_SIZE));
    document.querySelector("#history-page-label").textContent = `${data.total ? page : 0} / ${pages} trang`;
    document.querySelector("#history-prev").disabled = offset === 0;
    document.querySelector("#history-next").disabled = offset + PAGE_SIZE >= data.total;
    if (!data.items.length) {
      body.innerHTML = `<tr><td colspan="6" class="empty-cell">${data.total ? "Không có kết quả khớp bộ lọc." : "Chưa có ảnh nào được phân loại. Bắt đầu từ mục Nhận diện ảnh."}</td></tr>`;
      return;
    }
    body.innerHTML = data.items.map((item) => `<tr>
      <td class="filename-cell" title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</td>
      <td><span class="fruit-tag"><i></i>${escapeHtml(fruitLabel(item.fruit_type))}</span></td>
      <td>${escapeHtml(ripenessLabel(item.ripeness))}</td>
      <td>${item.confidence == null ? "—" : percent(item.confidence)}</td>
      <td>${Number(item.processing_time_ms || 0).toFixed(1)} ms</td>
      <td>${escapeHtml(formatDate(item.created_at))}</td>
    </tr>`).join("");
  } catch (error) {
    body.innerHTML = `<tr><td colspan="6" class="empty-cell">${escapeHtml(error.message)}</td></tr>`;
    document.querySelector("#history-count").textContent = "Không tải được";
  }
}

export async function mount() {
  const select = document.querySelector("#history-class");
  try {
    const info = await get("/api/model-info");
    select.insertAdjacentHTML("beforeend", (info.class_names || []).map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(classLabel(name))}</option>`).join(""));
  } catch { /* History vẫn có thể xem khi model tạm thời chưa nạp. */ }
  document.querySelector("#history-search").addEventListener("input", () => {
    window.clearTimeout(searchTimer);
    offset = 0;
    searchTimer = window.setTimeout(loadHistory, 220);
  });
  select.addEventListener("change", () => { offset = 0; loadHistory(); });
  document.querySelector("#history-prev").addEventListener("click", () => { offset = Math.max(0, offset - PAGE_SIZE); loadHistory(); });
  document.querySelector("#history-next").addEventListener("click", () => { offset += PAGE_SIZE; loadHistory(); });
  document.querySelector("#clear-history").addEventListener("click", async () => {
    if (!window.confirm("Xóa toàn bộ lịch sử phân loại trên máy này?")) return;
    try {
      await removeHistory();
      offset = 0;
      await loadHistory();
    } catch (error) {
      window.alert(error.message);
    }
  });
  await loadHistory();
}
