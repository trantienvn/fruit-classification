"""Chạy ứng dụng tại http://127.0.0.1:8000."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.web.main:app", host="127.0.0.1", port=8000)
