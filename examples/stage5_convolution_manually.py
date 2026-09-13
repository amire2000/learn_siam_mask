"""Compute a tiny valid 2D convolution with NumPy."""

import numpy as np


def convolve2d_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slide one 2D kernel over an image where it completely fits."""
    kernel_height, kernel_width = kernel.shape
    output_height = image.shape[0] - kernel_height + 1
    output_width = image.shape[1] - kernel_width + 1
    output = np.zeros((output_height, output_width), dtype=np.float32)

    for y in range(output_height):
        for x in range(output_width):
            patch = image[y:y + kernel_height, x:x + kernel_width]
            output[y, x] = np.sum(patch * kernel)

    return output


image = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
kernel = np.ones((2, 2), dtype=np.float32)
output = convolve2d_valid(image, kernel)
expected = np.array([[14, 18, 22], [30, 34, 38], [46, 50, 54]], dtype=np.float32)

assert np.array_equal(output, expected)

print("image:\n", image)
print("kernel:\n", kernel)
print("output:\n", output)
