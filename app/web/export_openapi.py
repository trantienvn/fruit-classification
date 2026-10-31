"""Xuất snapshot OpenAPI từ khai báo FastAPI hiện tại."""

from __future__ import annotations

import json
from pathlib import Path

from app.web.main import app


def main() -> None:
    output = Path(__file__).with_name("openapi.json")
    output.write_text(
        json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported OpenAPI contract: {output}")


if __name__ == "__main__":
    main()
