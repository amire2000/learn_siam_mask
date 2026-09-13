"""Apply the preprocessing steps from Stage 4 to a tiny BGR image."""

import cv2
import numpy as np


def preprocess(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Resize, scale, normalize, and convert one HWC image to CHW."""
    resized = cv2.resize(image, (2, 2))
    scaled = resized.astype(np.float32) / 255.0
    normalized = (scaled - mean) / std
    return normalized.transpose(2, 0, 1)


image = np.array(
    [
        [[0, 128, 255], [64, 64, 64]],
        [[255, 128, 0], [32, 192, 96]],
    ],
    dtype=np.uint8,
)
mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
std = np.array([0.25, 0.5, 0.25], dtype=np.float32)

chw = preprocess(image, mean, std)
first_pixel = chw[:, 0, 0]
expected = np.array([-2.0, (128 / 255 - 0.5) / 0.5, 2.0], dtype=np.float32)

assert chw.shape == (3, 2, 2)
assert chw.dtype == np.float32
assert np.allclose(first_pixel, expected, atol=1e-6)

print("input shape and dtype:", image.shape, image.dtype)
print("output shape and dtype:", chw.shape, chw.dtype)
print("first BGR pixel after mean/std normalization:", first_pixel)
