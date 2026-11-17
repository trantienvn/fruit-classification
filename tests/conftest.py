"""Fixtures kiểm thử đặt tệp tạm bên trong workspace."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Cung cấp thư mục riêng, không phụ thuộc thư mục Temp của Windows."""
    test_root = Path(__file__).resolve().parent / ".tmp"
    test_root.mkdir(exist_ok=True)
    path = test_root / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)