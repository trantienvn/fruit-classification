"""Trích xuất đặc trưng ảnh trái cây.

Đặc trưng trích xuất từ ảnh RGB crop (H×W×3, uint8):

  Color features (24):
    - Histogram RGB: 8 bins × 3 kênh = 24 giá trị (normalized)

  Color stats (9):
    - Mean và std của R, G, B (6), kênh H trong HSV (2), kênh S (2) → 6 + 3 = 9

  Texture — LBP (10):
    - Local Binary Patterns histogram (10 bins, normalized)

  Texture — GLCM (4):
    - contrast, dissimilarity, homogeneity, energy

  Shape (4):
    - area_ratio (diện tích bbox / tổng ảnh)
    - aspect_ratio (W/H)
    - extent (area / bounding rect area)
    - equivalent_diameter (normalized)

  Tổng: 24 + 9 + 10 + 4 + 4 = 51 features
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Số bins histogram màu
COLOR_HIST_BINS = 8
# LBP
LBP_RADIUS = 1
LBP_N_POINTS = 8
LBP_N_BINS = 10

FEATURE_NAMES: list[str] = (
    # RGB histogram
    [f"hist_r_{i}" for i in range(COLOR_HIST_BINS)]
    + [f"hist_g_{i}" for i in range(COLOR_HIST_BINS)]
    + [f"hist_b_{i}" for i in range(COLOR_HIST_BINS)]
    # Color stats
    + ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]
    + ["mean_h", "mean_s", "std_s"]
    # LBP
    + [f"lbp_{i}" for i in range(LBP_N_BINS)]
    # GLCM
    + ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy"]
    # Shape
    + ["shape_area_ratio", "shape_aspect_ratio", "shape_extent", "shape_eq_diameter"]
)

N_FEATURES = len(FEATURE_NAMES)  # 51


def _color_features(img_rgb: np.ndarray) -> np.ndarray:
    """24 + 9 = 33 đặc trưng màu sắc."""
    h, w = img_rgb.shape[:2]

    # Histogram RGB (normalized)
    hist_r = np.histogram(img_rgb[:, :, 0], bins=COLOR_HIST_BINS, range=(0, 256))[0]
    hist_g = np.histogram(img_rgb[:, :, 1], bins=COLOR_HIST_BINS, range=(0, 256))[0]
    hist_b = np.histogram(img_rgb[:, :, 2], bins=COLOR_HIST_BINS, range=(0, 256))[0]
    n_pixels = h * w
    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float64) / n_pixels

    # Mean & std RGB
    mean_rgb = img_rgb.mean(axis=(0, 1))  # (3,)
    std_rgb = img_rgb.std(axis=(0, 1))    # (3,)

    # Chuyển sang HSV để lấy hue và saturation
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mean_h = img_hsv[:, :, 0].mean() / 179.0   # normalize 0–1
    mean_s = img_hsv[:, :, 1].mean() / 255.0
    std_s = img_hsv[:, :, 1].std() / 255.0

    color_stats = np.array([
        mean_rgb[0] / 255, mean_rgb[1] / 255, mean_rgb[2] / 255,
        std_rgb[0] / 255, std_rgb[1] / 255, std_rgb[2] / 255,
        mean_h, mean_s, std_s,
    ])
    return np.concatenate([hist, color_stats])


def _lbp_features(img_rgb: np.ndarray) -> np.ndarray:
    """10 đặc trưng kết cấu LBP."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_N_POINTS + 2  # 10 với n_points=8
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    return hist.astype(np.float64) / (hist.sum() + 1e-9)


def _glcm_features(img_rgb: np.ndarray) -> np.ndarray:
    """4 đặc trưng GLCM."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Giảm mức xám về 64 để GLCM nhỏ gọn
    gray_q = (gray // 4).astype(np.uint8)
    glcm = graycomatrix(
        gray_q, distances=[1], angles=[0], levels=64,
        symmetric=True, normed=True,
    )
    contrast = graycoprops(glcm, "contrast")[0, 0]
    dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    # Normalize contrast và dissimilarity về xấp xỉ [0, 1]
    return np.array([
        contrast / 100.0,
        dissimilarity / 50.0,
        homogeneity,
        energy,
    ])


def _shape_features(img_rgb: np.ndarray) -> np.ndarray:
    """4 đặc trưng hình dạng từ mask ngưỡng Otsu."""
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Tìm contour lớn nhất
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(4, dtype=np.float64)

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    total_area = h * w
    x, y, bw, bh = cv2.boundingRect(cnt)

    area_ratio = area / total_area
    aspect_ratio = bw / (bh + 1e-9)
    extent = area / (bw * bh + 1e-9)
    eq_diameter = np.sqrt(4 * area / (np.pi * total_area + 1e-9))

    return np.array([area_ratio, aspect_ratio, extent, eq_diameter])


def extract_features(img_rgb: np.ndarray) -> np.ndarray:
    """Trích xuất vector 51 đặc trưng từ ảnh RGB uint8 (H×W×3).

    Sử dụng nhất quán trong cả training và inference.
    """
    assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3, "Cần ảnh RGB 3 kênh"
    color = _color_features(img_rgb)          # 33
    lbp = _lbp_features(img_rgb)              # 10
    glcm = _glcm_features(img_rgb)            # 4
    shape = _shape_features(img_rgb)          # 4
    feat = np.concatenate([color, lbp, glcm, shape])
    assert len(feat) == N_FEATURES, f"Expected {N_FEATURES} features, got {len(feat)}"
    return feat


def build_feature_matrix(
    images: list[np.ndarray],
) -> np.ndarray:
    """Tạo ma trận đặc trưng (n_samples, N_FEATURES) từ danh sách ảnh."""
    return np.array([extract_features(img) for img in images], dtype=np.float64)

