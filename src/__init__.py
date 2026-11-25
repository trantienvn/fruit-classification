"""
src — Nhánh học sâu (Deep Learning) cho dự án phân loại trái cây.

Pipeline: Ảnh gốc → CNN backbone (frozen) → embedding vector → scikit-learn classifier

Sử dụng MobileNetV3-Small pretrained (ImageNet) làm bộ trích xuất đặc trưng tự động.
Classifier là sklearn để dễ kiểm tra và so sánh.

Yêu cầu bổ sung:
    pip install torch torchvision
    # hoặc: pip install onnxruntime  (chỉ inference)

Thư mục artifact:
    data/processed/dl_features/   — cache CNN embeddings (.npz)
    models/                    — mô hình đã huấn luyện (.joblib + config.json)
    reports/                   — báo cáo đánh giá
"""

__version__ = "0.1.0"
__all__ = ["backbone", "preprocess", "train", "predict", "torch_classifier"]
