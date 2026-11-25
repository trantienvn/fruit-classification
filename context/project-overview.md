# Tổng quan dự án

- **Mục tiêu:** Phân loại ảnh trái cây bằng học sâu (Deep Learning); cung cấp API/UI để dự đoán.
- **Môi trường:** Python 3.14; PyTorch, torchvision, scikit-learn, Pillow, FastAPI và pytest.
- **Pipeline:** CNN backbone (MobileNetV3-Small, frozen) → embedding 576 chiều → MLP classifier. Huấn luyện GPU CUDA / CPU.
- **Dataset:** `data/processed/dataset/` — 9 lớp Cam/Chuối/Xoài × Chín/Chưa chín/Hỏng, 1.800 ảnh (200/lớp, split 70/15/15).
- **Ứng dụng:** `src/predict.py` cung cấp `FruitClassifierDL`; `app/web/` là FastAPI + HTML/CSS/JS; Edge Inference qua ONNX Runtime Web (`frontend/models/`).

Dataset và model/report artifacts là dữ liệu cục bộ; không thêm chúng vào Git.
