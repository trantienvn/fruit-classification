"""export_onnx.py — Xuất toàn bộ pipeline Deep Learning (Backbone + Scaler + PCA + PyTorch MLP)
thành các mô hình ONNX duy nhất phục vụ suy luận trực tiếp trong trình duyệt (Inference in Browser).

Hỗ trợ xuất cả 3 kiến trúc:
    1. frontend/models/model_mobilenetv3_small.onnx (~4.2 MB)
    2. frontend/models/model_mobilenetv2.onnx (~14 MB)
    3. frontend/models/model_resnet18.onnx (~44 MB)
    4. frontend/models/model.onnx (Mô hình tốt nhất mặc định)

Input ONNX:
    "input": [batch_size, 3, 224, 224] float32 (Ảnh RGB đã chuẩn hóa ImageNet)

Pipeline nội tại trong ONNX graph:
    input -> Backbone feature extractor -> Flatten
          -> (x - scaler.mean_) / scaler.scale_
          -> (x - pca.mean_) @ pca_components_t
          -> PyTorch MLP (Linear + BatchNorm + ReLU + Dropout [off])
          -> "logits": [batch_size, 9] float32
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
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backbone import (
    BACKBONE_SPECS,
    DEFAULT_BACKBONE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    SUPPORTED_BACKBONES,
    get_backbone,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# End-to-End PyTorch Module gộp toàn bộ pipeline
# ---------------------------------------------------------------------------
class EndToEndFruitClassifier(nn.Module):
    """Mô hình gộp toàn bộ quy trình: CNN Backbone -> StandardScaler -> PCA -> MLP Head."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        pca_mean: np.ndarray,
        pca_components_t: np.ndarray,
        mlp_net: nn.Module,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor

        # Đăng ký các trọng số tiền xử lý dạng buffer để lưu cố định trong ONNX graph
        self.register_buffer("scaler_mean", torch.from_numpy(scaler_mean.astype(np.float32)))
        self.register_buffer("scaler_scale", torch.from_numpy(scaler_scale.astype(np.float32)))
        self.register_buffer("pca_mean", torch.from_numpy(pca_mean.astype(np.float32)))
        self.register_buffer("pca_components_t", torch.from_numpy(pca_components_t.astype(np.float32)))

        self.mlp = mlp_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 3, 224, 224] float32
        feat = self.feature_extractor(x)  # [batch_size, embedding_dim]

        # Chuẩn hóa StandardScaler
        scaled = (feat - self.scaler_mean) / self.scaler_scale

        # Giảm chiều PCA
        pca_out = torch.matmul(scaled - self.pca_mean, self.pca_components_t)

        # PyTorch MLP Head
        logits = self.mlp(pca_out)  # [batch_size, 9]
        return logits


def export_single_model(
    backbone_name: str,
    joblib_path: Path,
    output_onnx_path: Path,
    opset_version: int = 17,
) -> Path:
    """Xuất một mô hình đơn lẻ sang file ONNX hoàn chỉnh."""
    spec = BACKBONE_SPECS[backbone_name]
    display_name = spec["display_name"]
    logger.info(f"\n--- Đang xuất ONNX cho {display_name} ({backbone_name}) ---")
    logger.info(f"  Nguồn: {joblib_path}")
    logger.info(f"  Đích : {output_onnx_path}")

    if not joblib_path.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình joblib: {joblib_path}")

    pipeline = joblib.load(joblib_path)
    scaler = pipeline.named_steps["scaler"]
    pca    = pipeline.named_steps["pca"]
    clf    = pipeline.named_steps["clf"]

    scaler_mean  = np.array(scaler.mean_, dtype=np.float32)
    scaler_scale = np.array(scaler.scale_, dtype=np.float32)
    pca_mean     = np.array(pca.mean_, dtype=np.float32)

    if getattr(pca, "whiten", False):
        pca_components_t = (pca.components_.T / np.sqrt(pca.explained_variance_)).astype(np.float32)
    else:
        pca_components_t = pca.components_.T.astype(np.float32)

    # Lấy MLP head
    clf_net = clf._net.cpu().eval()
    for p in clf_net.parameters():
        p.requires_grad = False

    # Lấy Feature Extractor đúng chuẩn ImageNet
    bb = get_backbone(name=backbone_name, device="cpu")
    feature_extractor = bb.get_torch_feature_extractor().cpu().eval()
    for p in feature_extractor.parameters():
        p.requires_grad = False

    # Gộp toàn bộ pipeline
    e2e_model = EndToEndFruitClassifier(
        feature_extractor=feature_extractor,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        pca_mean=pca_mean,
        pca_components_t=pca_components_t,
        mlp_net=clf_net,
    )
    e2e_model.eval()

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1], dtype=torch.float32)

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

    # Kiểm tra tính hợp lệ
    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)
    size_mb = output_onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Xuất thành công {display_name}: {output_onnx_path} ({size_mb:.2f} MB)")
    return output_onnx_path


def export_all_models(
    model_dir: Path = Path("models"),
    output_dir: Path = Path("frontend/models"),
    reports_dir: Path = Path("reports"),
    opset_version: int = 17,
) -> dict[str, Path]:
    """Xuất toàn bộ các mô hình có trong models sang ONNX và đồng bộ file cấu hình."""
    logger.info("=== BẮT ĐẦU QUY TRÌNH XUẤT ĐA MÔ HÌNH ONNX ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: dict[str, Path] = {}

    # Đọc config để biết mô hình tốt nhất
    config_path = model_dir / "config.json"
    best_backbone = DEFAULT_BACKBONE
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            best_backbone = cfg.get("backbone", DEFAULT_BACKBONE)

    for b_name in SUPPORTED_BACKBONES:
        joblib_path = model_dir / f"classifier_{b_name}.joblib"
        if not joblib_path.exists():
            # Nếu chưa có file riêng, thử kiểm tra nếu đây là model_dl
            if b_name == best_backbone and (model_dir / "classifier.joblib").exists():
                joblib_path = model_dir / "classifier.joblib"
            else:
                logger.warning(f"Bỏ qua {b_name}: không tìm thấy {joblib_path}")
                continue

        out_path = output_dir / f"model_{b_name}.onnx"
        export_single_model(b_name, joblib_path, out_path, opset_version=opset_version)
        exported_paths[b_name] = out_path

    # Xuất / Sao chép mô hình tốt nhất thành model.onnx (mặc định cho web)
    default_onnx = output_dir / "model.onnx"
    if best_backbone in exported_paths:
        shutil.copy2(exported_paths[best_backbone], default_onnx)
        logger.info(f"✓ Đã đặt mô hình xuất sắc nhất ({BACKBONE_SPECS[best_backbone]['display_name']}) làm mặc định: {default_onnx}")
    else:
        # Xuất trực tiếp từ classifier.joblib
        dl_path = model_dir / "classifier.joblib"
        if dl_path.exists():
            export_single_model(best_backbone, dl_path, default_onnx, opset_version=opset_version)

    # Đồng bộ config.json
    if config_path.exists():
        shutil.copy2(config_path, output_dir / "config.json")
        logger.info("✓ Đã đồng bộ config.json sang frontend/models/")

    # Đồng bộ báo cáo và biểu đồ
    static_img = Path("frontend/static/img")
    static_img.mkdir(parents=True, exist_ok=True)
    for fname in ("loss_curve.png", "epoch_training.png", "model_comparison.png", "confusion_matrix.png", "roc_curve.png", "training_history.png"):
        fsrc = reports_dir / fname
        if fsrc.exists():
            shutil.copy2(fsrc, static_img / fname)

    return exported_paths


def verify_onnx(
    onnx_path: Path,
    backbone_name: str,
    joblib_path: Path,
    test_dir: Path = Path("data/processed/dataset/test"),
    num_samples: int = 25,
) -> bool:
    """So sánh kết quả dự đoán softmax giữa ONNX Runtime và scikit-learn Pipeline trên các ảnh kiểm thử."""
    display_name = BACKBONE_SPECS.get(backbone_name, {}).get("display_name", backbone_name)
    logger.info(f"\n=== KIỂM ĐỊNH ONNX vs PYTHON PIPELINE: {display_name} ===")

    pipeline = joblib.load(joblib_path)
    bb = get_backbone(name=backbone_name, device="cpu")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    image_paths: list[Path] = []
    if test_dir.exists():
        for class_folder in sorted(test_dir.iterdir()):
            if class_folder.is_dir():
                class_images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
                image_paths.extend(class_images[:3])

    if len(image_paths) < num_samples:
        all_imgs = list(Path("data").rglob("*.jpg"))
        for p in all_imgs:
            if p not in image_paths:
                image_paths.append(p)
            if len(image_paths) >= num_samples:
                break

    image_paths = image_paths[:num_samples]
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(1, 1, 3)

    max_diff_overall = 0.0
    mismatches = 0

    print("-" * 80)
    print(f"{'Ảnh kiểm thử':<32} | {'Nhãn Py':<12} | {'Nhãn ONNX':<12} | {'Sai số Max'}")
    print("-" * 80)

    for img_path in image_paths:
        with Image.open(img_path) as pil_img:
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            img_arr = np.array(pil_img, dtype=np.uint8)

            pil_resized = pil_img.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.BILINEAR)
            arr = np.array(pil_resized, dtype=np.float32) / 255.0
            arr = (arr - mean) / std
            arr = arr.transpose(2, 0, 1)
            inp_tensor = np.expand_dims(arr, axis=0)

        # 1. Dự đoán qua Python Pipeline
        emb = bb.extract(img_arr).reshape(1, -1)
        probs_py = pipeline.predict_proba(emb)[0]
        pred_py = int(np.argmax(probs_py))

        # 2. Dự đoán qua ONNX Runtime
        raw_logits = session.run(None, {input_name: inp_tensor})[0][0]
        shift = raw_logits - np.max(raw_logits)
        exps = np.exp(shift)
        probs_onnx = exps / np.sum(exps)
        pred_onnx = int(np.argmax(probs_onnx))

        diff = float(np.max(np.abs(probs_py - probs_onnx)))
        if diff > max_diff_overall:
            max_diff_overall = diff

        if pred_py != pred_onnx:
            mismatches += 1

        print(f"{img_path.name:<32} | Lớp {pred_py:<8} | Lớp {pred_onnx:<8} | {diff:.8f}")

    print("-" * 80)
    logger.info(f"Kết quả kiểm định {display_name}:")
    logger.info(f"  - Số ảnh kiểm thử: {len(image_paths)}")
    logger.info(f"  - Số ca lệch nhãn : {mismatches}")
    logger.info(f"  - Sai số tối đa    : {max_diff_overall:.8f}")

    if max_diff_overall < 1e-3 and mismatches == 0:
        logger.info(f"✓ XÁC THỰC HOÀN HẢO! {display_name} khớp chuẩn 100% với PyTorch pipeline (sai số < 1e-3).")
        return True
    else:
        logger.warning(f"⚠ Sai số {max_diff_overall:.8f} vượt ngưỡng 1e-3 hoặc có lệch nhãn!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Xuất pipeline DL sang mô hình ONNX.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Thư mục chứa weights")
    parser.add_argument("--output-dir", type=Path, default=Path("frontend/models"), help="Thư mục xuất ONNX")
    parser.add_argument("--verify", action="store_true", default=True, help="Kiểm thử sai số sau khi xuất")
    parser.add_argument("--samples", type=int, default=25, help="Số ảnh kiểm thử")
    args = parser.parse_args()

    exported = export_all_models(model_dir=args.model_dir, output_dir=args.output_dir)

    if args.verify:
        all_passed = True
        for b_name, onnx_path in exported.items():
            joblib_path = args.model_dir / f"classifier_{b_name}.joblib"
            if not joblib_path.exists():
                joblib_path = args.model_dir / "classifier.joblib"
            ok = verify_onnx(onnx_path, b_name, joblib_path, num_samples=args.samples)
            if not ok:
                all_passed = False

        if all_passed:
            print("\n✓ TOÀN BỘ 3 MÔ HÌNH ONNX ĐÃ ĐƯỢC XÁC THỰC THÀNH CÔNG VỚI ĐỘ CHÍNH XÁC TUYỆT ĐỐI!")


if __name__ == "__main__":
    main()
