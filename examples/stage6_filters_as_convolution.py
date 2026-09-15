"""Use hand-written convolution filters on a simple black-and-white image."""

import numpy as np


def convolve2d_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply one kernel everywhere it completely fits in a grayscale image."""
    kernel_height, kernel_width = kernel.shape
    output_height = image.shape[0] - kernel_height + 1
    output_width = image.shape[1] - kernel_width + 1
    output = np.zeros((output_height, output_width), dtype=np.float32)

    for y in range(output_height):
        for x in range(output_width):
            patch = image[y:y + kernel_height, x:x + kernel_width]
            output[y, x] = np.sum(patch * kernel)

    return output


image = np.zeros((7, 7), dtype=np.float32)
image[:, 3:] = 1.0

blur_kernel = np.ones((3, 3), dtype=np.float32) / 9.0
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
vertical_edge_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)

blurred = convolve2d_valid(image, blur_kernel)
sharpened = convolve2d_valid(image, sharpen_kernel)
edges = convolve2d_valid(image, vertical_edge_kernel)

assert blurred.shape == (5, 5)
assert np.allclose(edges[0], [0, 3, 3, 0, 0])
assert np.allclose(edges[0, [0, 3, 4]], 0)

print("input:\n", image)
print("blurred:\n", blurred)
print("sharpened:\n", sharpened)
print("vertical-edge response:\n", edges)
