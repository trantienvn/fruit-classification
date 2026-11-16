"""export_onnx.py — Xuất toàn bộ pipeline Deep Learning (MobileNetV3 + Scaler + PCA + PyTorch MLP)
thành MỘT mô hình ONNX duy nhất phục vụ suy luận trực tiếp trong trình duyệt (Inference in Browser).

Input ONNX:
    "input": [batch_size, 3, 224, 224] float32 (Ảnh RGB đã chuẩn hóa ImageNet)

Pipeline nội tại trong ONNX graph:
    input -> MobileNetV3-Small features -> AdaptiveAvgPool2d -> Flatten(576-D)
          -> (x - scaler.mean_) / scaler.scale_
          -> (x - pca.mean_) @ pca_components_t
          -> PyTorch MLP (Linear + BatchNorm + ReLU + Dropout [off])
          -> "logits": [batch_size, 9] float32

Đầu ra:
    frontend/models/model.onnx
    frontend/models/config.json
    frontend/static/img/loss_curve.png, epoch_training.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

# Đảm bảo import được src_dl
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_dl.backbone import (
    BACKBONE_NAME,
    EMBEDDING_DIM,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
)
from src_dl.predict_dl import FruitClassifierDL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# End-to-End PyTorch Module gộp toàn bộ pipeline
# ---------------------------------------------------------------------------
class EndToEndFruitClassifier(nn.Module):
    """Mô hình gộp toàn bộ quy trình: CNN Backbone -> StandardScaler -> PCA -> MLP Head."""

    def __init__(
        self,
        features: nn.Module,
        avgpool: nn.Module,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        pca_mean: np.ndarray,
        pca_components_t: np.ndarray,
        mlp_net: nn.Module,
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = avgpool

        # Đăng ký các trọng số tiền xử lý dạng buffer để lưu cố định trong ONNX graph
        self.register_buffer("scaler_mean", torch.from_numpy(scaler_mean.astype(np.float32)))
        self.register_buffer("scaler_scale", torch.from_numpy(scaler_scale.astype(np.float32)))
        self.register_buffer("pca_mean", torch.from_numpy(pca_mean.astype(np.float32)))
        self.register_buffer("pca_components_t", torch.from_numpy(pca_components_t.astype(np.float32)))

        self.mlp = mlp_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 3, 224, 224] float32
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)  # [batch_size, 576]

        # Chuẩn hóa StandardScaler
        scaled = (feat - self.scaler_mean) / self.scaler_scale

        # Giảm chiều PCA
        pca_out = torch.matmul(scaled - self.pca_mean, self.pca_components_t)

        # PyTorch MLP Head
        logits = self.mlp(pca_out)  # [batch_size, 9]
        return logits


def export_model(
    model_dir: Path = Path("models_dl"),
    output_dir: Path = Path("frontend/models"),
    static_img_dir: Path = Path("frontend/static/img"),
    reports_dir: Path = Path("reports_dl"),
    opset_version: int = 17,
) -> Path:
    """Đọc pipeline từ joblib, xây dựng mô hình gộp và xuất ra frontend/models/model.onnx."""
    logger.info("=== BẮT ĐẦU QUY TRÌNH XUẤT ONNX MODEL ===")

    clf_path = model_dir / "classifier_dl.joblib"
    config_path = model_dir / "config.json"

    if not clf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình: {clf_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy cấu hình: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    logger.info("1. Nạp pipeline scikit-learn & PyTorch từ %s", clf_path)
    pipeline = joblib.load(clf_path)

    # Kiểm tra cấu trúc pipeline
    if not hasattr(pipeline, "named_steps"):
        raise ValueError(f"Mô hình nạp được không phải là Pipeline sklearn: {type(pipeline)}")

    scaler = pipeline.named_steps["scaler"]
    pca = pipeline.named_steps["pca"]
    clf = pipeline.named_steps["clf"]

    scaler_mean = np.array(scaler.mean_, dtype=np.float32)
    scaler_scale = np.array(scaler.scale_, dtype=np.float32)
    pca_mean = np.array(pca.mean_, dtype=np.float32)

    # Xử lý PCA components (xét cả trường hợp whiten=True/False)
    if getattr(pca, "whiten", False):
        logger.info("   PCA sử dụng whiten=True -> chuẩn hóa components theo explained_variance_")
        pca_components_t = (pca.components_.T / np.sqrt(pca.explained_variance_)).astype(np.float32)
    else:
        logger.info("   PCA sử dụng whiten=False -> dùng trực tiếp components_.T")
        pca_components_t = pca.components_.T.astype(np.float32)

    logger.info("   Scaler mean shape: %s | Scale shape: %s", scaler_mean.shape, scaler_scale.shape)
    logger.info("   PCA mean shape: %s | Components.T shape: %s", pca_mean.shape, pca_components_t.shape)

    # Đưa PyTorch MLP về CPU và eval mode
    clf_net = clf._net.cpu()
    clf_net.eval()
    for p in clf_net.parameters():
        p.requires_grad = False

    logger.info("2. Khởi tạo MobileNetV3-Small Backbone đúng chuẩn ImageNet pretrained")
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    full_mobilenet = models.mobilenet_v3_small(weights=weights)
    features = full_mobilenet.features.cpu().eval()
    avgpool = full_mobilenet.avgpool.cpu().eval()
    for p in features.parameters():
        p.requires_grad = False
    for p in avgpool.parameters():
        p.requires_grad = False

    logger.info("3. Ghép nối EndToEndFruitClassifier")
    e2e_model = EndToEndFruitClassifier(
        features=features,
        avgpool=avgpool,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        pca_mean=pca_mean,
        pca_components_t=pca_components_t,
        mlp_net=clf_net,
    )
    e2e_model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_onnx_path = output_dir / "model.onnx"

    logger.info("4. Xuất file ONNX: %s (opset=%d, single-file)", output_onnx_path, opset_version)
    dummy_input = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1], dtype=torch.float32)

    # Đảm bảo xuất ra 1 file ONNX duy nhất
    torch.onnx.export(
        e2e_model,
        dummy_input,
        str(output_onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )

    # Kiểm tra ONNX model hợp lệ
    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("✓ File ONNX hợp lệ theo tiêu chuẩn onnx.checker!")

    # Kiểm tra dung lượng
    file_size_mb = output_onnx_path.stat().st_size / (1024 * 1024)
    logger.info("✓ Dung lượng file ONNX: %.2f MB", file_size_mb)
    if file_size_mb > 30.0:
        logger.warning("⚠ CẢNH BÁO: Kích thước mô hình > 30MB (%.2f MB). Hãy cân nhắc tối ưu!", file_size_mb)
    else:
        logger.info("✓ Kích thước mô hình đạt chuẩn lý tưởng cho trình duyệt (< 30 MB).")

    # Sao chép config.json sang frontend/models/
    target_config = output_dir / "config.json"
    shutil.copy2(config_path, target_config)
    logger.info("✓ Đã sao chép config.json sang: %s", target_config)

    # Sao chép báo cáo sang frontend/static/img/
    static_img_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("loss_curve.png", "epoch_training.png"):
        src_file = reports_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, static_img_dir / fname)
            logger.info("✓ Đã sao chép %s sang: %s", fname, static_img_dir)

    return output_onnx_path


def verify_onnx(
    onnx_path: Path,
    model_dir: Path = Path("models_dl"),
    test_dir: Path = Path("data/processed/dataset/test"),
    num_samples: int = 25,
) -> None:
    """So sánh kết quả dự đoán softmax giữa ONNX Runtime và FruitClassifierDL Python trên >= 20 ảnh."""
    logger.info("\n=== BẮT ĐẦU KIỂM ĐỊNH (VERIFICATION) ONNX vs PYTHON PIPELINE ===")

    # 1. Khởi tạo Python FruitClassifierDL
    clf_py = FruitClassifierDL(model_dir=model_dir, device="cpu")
    clf_py.load()

    # 2. Khởi tạo ONNX Runtime Session
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # 3. Thu thập danh sách ảnh kiểm thử từ các thư mục lớp
    image_paths: list[Path] = []
    if test_dir.exists():
        for class_folder in sorted(test_dir.iterdir()):
            if class_folder.is_dir():
                class_images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
                image_paths.extend(class_images[:3])  # Lấy 3 ảnh mỗi lớp

    if len(image_paths) < 20:
        # Nếu chưa đủ 20 ảnh, tìm thêm trong toàn bộ data
        all_imgs = list(Path("data").rglob("*.jpg"))
        for p in all_imgs:
            if p not in image_paths:
                image_paths.append(p)
            if len(image_paths) >= num_samples:
                break

    if len(image_paths) < num_samples:
        logger.warning("Không tìm đủ %d ảnh thực tế, chỉ tìm thấy %d ảnh.", num_samples, len(image_paths))
    else:
        image_paths = image_paths[:num_samples]

    logger.info("Tiến hành kiểm thử trên %d ảnh mẫu...", len(image_paths))

    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    max_diff_overall = 0.0
    mismatches = 0

    print("-" * 85)
    print(f"{'Ảnh kiểm thử':<35} | {'Dự đoán Py':<15} | {'Dự đoán ONNX':<15} | {'Sai số Max'}")
    print("-" * 85)

    for img_path in image_paths:
        # Dự đoán qua FruitClassifierDL (Python pipeline)
        res_py = clf_py.predict_from_path(img_path)
        probs_py = np.array([res_py["probabilities"][cls] for cls in clf_py.class_names], dtype=np.float32)
        pred_py = res_py["class_name"]

        # Dự đoán qua ONNX Runtime
        with Image.open(img_path) as pil_img:
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_resized = pil_img.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.BILINEAR)
            arr = np.array(pil_resized, dtype=np.float32) / 255.0
            arr = (arr - mean) / std
            arr = arr.transpose(2, 0, 1)  # (3, 224, 224)
            inp_tensor = np.expand_dims(arr, axis=0)  # (1, 3, 224, 224)

        raw_logits = session.run(None, {input_name: inp_tensor})[0][0]

        # Softmax ổn định số học
        shift_logits = raw_logits - np.max(raw_logits)
        exps = np.exp(shift_logits)
        probs_onnx = exps / np.sum(exps)

        pred_idx_onnx = int(np.argmax(probs_onnx))
        pred_onnx = clf_py.class_names[pred_idx_onnx]

        diff = float(np.max(np.abs(probs_py - probs_onnx)))
        if diff > max_diff_overall:
            max_diff_overall = diff

        if pred_py != pred_onnx:
            mismatches += 1

        print(f"{img_path.name:<35} | {pred_py:<15} | {pred_onnx:<15} | {diff:.8f}")

    print("-" * 85)
    logger.info("KẾT QUẢ KIỂM ĐỊNH:")
    logger.info("  - Tổng số ảnh kiểm thử: %d", len(image_paths))
    logger.info("  - Số ca lệch nhãn phân loại: %d", mismatches)
    logger.info("  - Sai số xác suất tuyệt đối tối đa (Max Diff): %.8f", max_diff_overall)

    if max_diff_overall < 1e-3:
        logger.info("✓ TUYỆT VỜI! Sai số tối đa %.8f < 1e-3 (Khớp chuẩn 100%% với Python pipeline)!", max_diff_overall)
    else:
        logger.warning("⚠ Sai số %.8f vượt ngưỡng kỳ vọng 1e-3, cần kiểm tra lại!", max_diff_overall)


def main():
    parser = argparse.ArgumentParser(description="Xuất pipeline DL sang mô hình ONNX duy nhất.")
    parser.add_argument("--model-dir", type=Path, default=Path("models_dl"), help="Thư mục chứa weights")
    parser.add_argument("--output-dir", type=Path, default=Path("frontend/models"), help="Thư mục xuất ONNX")
    parser.add_argument("--verify", action="store_true", default=True, help="Kiểm thử sai số sau khi xuất")
    parser.add_argument("--samples", type=int, default=25, help="Số ảnh kiểm thử")
    args = parser.parse_args()

    onnx_path = export_model(model_dir=args.model_dir, output_dir=args.output_dir)
    if args.verify:
        verify_onnx(onnx_path, model_dir=args.model_dir, num_samples=args.samples)


if __name__ == "__main__":
    main()
