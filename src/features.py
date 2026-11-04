"""Trích xuất đặc trưng màu, kết cấu và hình dạng trong mask vùng quả."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

COLOR_HIST_BINS = 8
LBP_RADIUS = 1
LBP_N_POINTS = 8
LBP_N_BINS = 10

COLOR_FEATURE_NAMES = (
  [f"hist_rgb_{channel}_{index}" for channel in "rgb" for index in range(COLOR_HIST_BINS)]
  + [f"hist_hsv_{channel}_{index}" for channel in "hsv" for index in range(COLOR_HIST_BINS)]
  + [f"{stat}_{channel}" for space, channels in (("rgb", "rgb"), ("hsv", "hsv"), ("lab", "lab"))
     for stat in ("mean", "std") for channel in channels]
  + ["ratio_green", "ratio_yellow", "ratio_orange", "ratio_brown_black"]
)
TEXTURE_FEATURE_NAMES = (
  [f"lbp_{index}" for index in range(LBP_N_BINS)]
  + ["glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]
)
SHAPE_FEATURE_NAMES = [
  "shape_area_ratio", "shape_perimeter_ratio", "shape_width_ratio",
  "shape_height_ratio", "shape_aspect_ratio", "shape_circularity",
]
FEATURE_NAMES: list[str] = COLOR_FEATURE_NAMES + TEXTURE_FEATURE_NAMES + SHAPE_FEATURE_NAMES
N_FEATURES = len(FEATURE_NAMES)
FEATURE_GROUPS = {
  "color": slice(0, len(COLOR_FEATURE_NAMES)),
  "texture": slice(len(COLOR_FEATURE_NAMES), len(COLOR_FEATURE_NAMES) + len(TEXTURE_FEATURE_NAMES)),
  "shape": slice(len(COLOR_FEATURE_NAMES) + len(TEXTURE_FEATURE_NAMES), N_FEATURES),
}


def segment_fruit(img_rgb: np.ndarray) -> np.ndarray:
  """Tách vùng quả bằng Otsu trên S/xám, morphology và adaptive fallback."""
  denoised = cv2.medianBlur(img_rgb, 3)
  gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
  saturation = cv2.cvtColor(denoised, cv2.COLOR_RGB2HSV)[:, :, 1]
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  candidates = []

  for channel in (saturation, gray):
    _, threshold = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for binary in (threshold, cv2.bitwise_not(threshold)):
      cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
      cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
      contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      if contours:
        largest = max(contours, key=cv2.contourArea)
        component = np.zeros_like(cleaned)
        cv2.drawContours(component, [largest], -1, 255, thickness=cv2.FILLED)
        ratio = float(np.count_nonzero(component)) / component.size
        if 0.05 <= ratio <= 0.95:
          candidates.append((ratio, component))

  if candidates:
    _, mask = min(candidates, key=lambda item: abs(item[0] - 0.5))
    return mask

  adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
  )
  adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
  contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if contours:
    mask = np.zeros_like(adaptive)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    ratio = np.count_nonzero(mask) / mask.size
    if 0.05 <= ratio <= 0.95:
      return mask
  return np.full(gray.shape, 255, dtype=np.uint8)


def _color_features(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
  pixels_rgb = img_rgb[mask > 0]
  if not len(pixels_rgb):
    pixels_rgb = img_rgb.reshape(-1, 3)
  hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[mask > 0]
  lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)[mask > 0]
  if not len(hsv):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3)

  histograms = []
  for values, bounds in ((pixels_rgb, (256, 256, 256)), (hsv, (180, 256, 256))):
    for channel in range(3):
      histograms.extend(
        np.histogram(values[:, channel], bins=COLOR_HIST_BINS, range=(0, bounds[channel]))[0]
        / len(values)
      )
  stats = []
  for values, scale in ((pixels_rgb, 255.0), (hsv, 255.0), (lab, 255.0)):
    stats.extend(values.mean(axis=0) / scale)
    stats.extend(values.std(axis=0) / scale)

  hue, _, value = hsv.T
  ratios = [
    np.mean((hue >= 30) & (hue <= 85)),
    np.mean((hue >= 15) & (hue < 35)),
    np.mean((hue >= 5) & (hue < 25)),
    np.mean((value < 80) | ((hue >= 5) & (hue < 25) & (value < 150))),
  ]
  return np.asarray([*histograms, *stats, *ratios], dtype=np.float64)


def _texture_features(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
  gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
  lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method="uniform")
  selected = lbp[mask > 0]
  hist, _ = np.histogram(selected, bins=LBP_N_BINS, range=(0, LBP_N_BINS))
  lbp_hist = hist.astype(np.float64) / (hist.sum() + 1e-9)

  ys, xs = np.where(mask > 0)
  if not len(xs):
    return np.concatenate([lbp_hist, np.zeros(4)])
  x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
  crop = gray[y1:y2, x1:x2].copy()
  crop_mask = mask[y1:y2, x1:x2] > 0
  crop[~crop_mask] = int(np.median(crop[crop_mask]))
  gray_q = (crop // 4).astype(np.uint8)
  glcm = graycomatrix(
    gray_q, distances=[1], angles=[0], levels=64, symmetric=True, normed=True
  )
  properties = [
    graycoprops(glcm, name)[0, 0]
    for name in ("contrast", "homogeneity", "energy", "correlation")
  ]
  properties[0] /= 100.0
  return np.concatenate([lbp_hist, np.nan_to_num(properties)])


def _shape_features(mask: np.ndarray) -> np.ndarray:
  height, width = mask.shape
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if not contours:
    return np.zeros(len(SHAPE_FEATURE_NAMES), dtype=np.float64)
  contour = max(contours, key=cv2.contourArea)
  area = cv2.contourArea(contour)
  perimeter = cv2.arcLength(contour, True)
  x, y, box_width, box_height = cv2.boundingRect(contour)
  return np.asarray([
    area / (height * width),
    perimeter / np.sqrt(height * width),
    box_width / width,
    box_height / height,
    box_width / (box_height + 1e-9),
    4 * np.pi * area / (perimeter * perimeter + 1e-9),
  ])


def extract_features(img_rgb: np.ndarray) -> np.ndarray:
    """Trích xuất vector 90 đặc trưng từ ảnh RGB uint8 (H×W×3).

    Sử dụng nhất quán trong cả training và inference.
    """
    assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3, "Cần ảnh RGB 3 kênh"
    denoised = cv2.medianBlur(img_rgb, 3)
    mask = segment_fruit(denoised)
    color = _color_features(denoised, mask)
    texture = _texture_features(denoised, mask)
    shape = _shape_features(mask)
    feat = np.concatenate([color, texture, shape])
    assert len(feat) == N_FEATURES, f"Expected {N_FEATURES} features, got {len(feat)}"
    return feat


def save_mask_samples(
    images: list[np.ndarray],
    save_dir: str,
    labels: list[int] | np.ndarray | None = None,
) -> int:
    """Lưu mask train, ưu tiên một mẫu cho mỗi lớp để kiểm tra phân đoạn."""
    from pathlib import Path

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    seen_labels = set()
    if labels is not None:
        for image, label in zip(images, labels):
            if int(label) not in seen_labels:
                seen_labels.add(int(label))
                selected.append(image)
            if len(selected) == 9:
                break
    else:
        selected = images[:9]
    for index, image in enumerate(selected):
        cv2.imwrite(str(output_dir / f"mask_{index:02d}.png"), segment_fruit(image))
    return len(selected)


def augment_training_images(
    images: list[np.ndarray],
    labels: list[int] | np.ndarray,
    random_state: int = 42,
) -> tuple[list[np.ndarray], list[int]]:
    """Tạo một ảnh xoay/lật/đổi sáng cho mỗi mẫu train; không nhận val/test."""
    rng = np.random.default_rng(random_state)
    augmented_images = list(images)
    augmented_labels = [int(label) for label in labels]
    for image, label in zip(images, labels):
        height, width = image.shape[:2]
        angle = float(rng.uniform(-15, 15))
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        variant = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
        )
        if rng.random() < 0.5:
            variant = cv2.flip(variant, 1)
        factor = float(rng.uniform(0.85, 1.15))
        offset = float(rng.uniform(-12, 12))
        variant = np.clip(variant.astype(np.float32) * factor + offset, 0, 255).astype(np.uint8)
        augmented_images.append(variant)
        augmented_labels.append(int(label))
    return augmented_images, augmented_labels


def build_feature_matrix(
    images: list[np.ndarray],
) -> np.ndarray:
    """Tạo ma trận đặc trưng (n_samples, N_FEATURES) từ danh sách ảnh."""
    return np.array([extract_features(img) for img in images], dtype=np.float64)

