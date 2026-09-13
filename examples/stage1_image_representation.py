"""Inspect a tiny RGB image tensor."""

import numpy as np


image = np.array(
    [
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[255, 255, 255], [0, 0, 0], [128, 128, 128]],
    ],
    dtype=np.uint8,
)

print("shape:", image.shape)
print("top-left pixel (RGB):", image[0, 0])
print("dtype:", image.dtype)
print("normalized range:", (image / 255.0).min(), "to", (image / 255.0).max())
