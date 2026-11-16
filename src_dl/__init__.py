"""
src_dl — Nhánh học sâu (Deep Learning) cho dự án phân loại trái cây.

Pipeline: Ảnh gốc → CNN backbone (frozen) → embedding vector → scikit-learn classifier

Khác với src/ (đặc trưng thủ công / pixel phẳng), src_dl/ sử dụng
MobileNetV3-Small pretrained (ImageNet) làm bộ trích xuất đặc trưng tự động.
Classifier vẫn là sklearn để dễ so sánh với nhánh src/.

Yêu cầu bổ sung:
    pip install torch torchvision
    # hoặc: pip install onnxruntime  (chỉ inference)

Thư mục artifact:
    data/processed/dl_features/   — cache CNN embeddings (.npz)
    models_dl/                    — mô hình đã huấn luyện (.joblib + config.json)
    reports_dl/                   — báo cáo đánh giá
"""

__version__ = "0.1.0"
__all__ = ["backbone", "preprocess_dl", "train_dl", "predict_dl"]
