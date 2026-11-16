"""Kiểm thử tự động cho file mô hình ONNX đã xuất (frontend/models/model.onnx).
Xác thực ONNX graph, kiểm tra đầu vào, đầu ra, và so sánh với config.json.
"""

import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest


def test_onnx_model_exists_and_valid():
    onnx_path = Path("frontend/models/model.onnx")
    config_path = Path("frontend/models/config.json")

    assert onnx_path.exists(), "Tệp frontend/models/model.onnx không tồn tại!"
    assert config_path.exists(), "Tệp frontend/models/config.json không tồn tại!"

    # 1. Kiểm tra kích thước (< 30 MB)
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Kích thước file ONNX: {size_mb:.2f} MB")
    assert size_mb < 30.0, f"Kích thước {size_mb:.2f} MB vượt quá ngưỡng 30MB"

    # 2. Kiểm tra ONNX graph
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    # 3. Kiểm tra input / output shapes
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    assert len(inputs) == 1, "Mô hình phải có đúng 1 input"
    assert inputs[0].name == "input", f"Tên input phải là 'input', nhận được {inputs[0].name}"
    assert inputs[0].shape[1:] == [3, 224, 224], f"Input shape phải là [batch, 3, 224, 224], nhận được {inputs[0].shape}"

    assert len(outputs) == 1, "Mô hình phải có đúng 1 output"
    assert outputs[0].name == "logits", f"Tên output phải là 'logits', nhận được {outputs[0].name}"
    assert outputs[0].shape[1] == 9, f"Output shape phải là [batch, 9], nhận được {outputs[0].shape}"

    # 4. Chạy thử suy luận dummy
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    raw_logits = session.run(None, {"input": dummy_input})[0]
    assert raw_logits.shape == (1, 9)

    # 5. Kiểm tra config.json
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert len(cfg.get("class_names", [])) == 9
    assert cfg.get("crop_size") == [224, 224]
    print("✓ Kiểm tra ONNX model và cấu hình thành công!")


if __name__ == "__main__":
    test_onnx_model_exists_and_valid()
