"""Kiểm thử tự động cho các file mô hình ONNX đã xuất trong frontend/models/.
Xác thực ONNX graph, kiểm tra đầu vào, đầu ra, và chạy thử dummy inference.
"""

import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest


@pytest.mark.parametrize("model_filename, max_size_mb", [
    ("model.onnx", 50.0),
    ("model_mobilenetv3_small.onnx", 10.0),
    ("model_mobilenetv2.onnx", 20.0),
    ("model_resnet18.onnx", 50.0),
])
def test_onnx_models_validity(model_filename: str, max_size_mb: float):
    onnx_path = Path("frontend/models") / model_filename
    config_path = Path("frontend/models/config.json")

    assert onnx_path.exists(), f"Tệp {onnx_path} không tồn tại!"
    assert config_path.exists(), "Tệp frontend/models/config.json không tồn tại!"

    # 1. Kiểm tra kích thước file
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"[{model_filename}] Kích thước file ONNX: {size_mb:.2f} MB")
    assert size_mb < max_size_mb, f"Kích thước {size_mb:.2f} MB vượt quá ngưỡng {max_size_mb} MB"

    # 2. Kiểm tra ONNX graph theo chuẩn onnx.checker
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    # 3. Khởi tạo session và kiểm tra input / output shapes
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


if __name__ == "__main__":
    for m, s in [("model.onnx", 50.0), ("model_mobilenetv3_small.onnx", 10.0), ("model_mobilenetv2.onnx", 20.0), ("model_resnet18.onnx", 50.0)]:
        test_onnx_models_validity(m, s)
    print("✓ Toàn bộ các file ONNX kiểm tra hợp lệ và đạt chuẩn!")

