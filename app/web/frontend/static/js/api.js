export class ApiError extends Error {
  constructor(message, status, code = "request_error") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    throw new ApiError("Không thể kết nối tới API. Hãy kiểm tra máy chủ.", 0, "network_error");
  }

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const error = body?.error;
    throw new ApiError(
      error?.message || `API trả về lỗi ${response.status}.`,
      response.status,
      error?.code || "request_error",
    );
  }
  return body?.data;
}

export function get(path, params = {}) {
  const url = new URL(path, window.location.origin);
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, String(value));
    }
  }
  return request(url.pathname + url.search);
}

export function postImage(file) {
  const body = new FormData();
  body.append("file", file, file.name);
  return request("/api/predict", { method: "POST", body });
}

export function postTabular(features) {
  return request("/api/tabular/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
  });
}

export function removeHistory() {
  return request("/api/history", { method: "DELETE" });
}
