"""Practice the NumPy image operations used in Stage 2."""

import numpy as np


image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)

crop = image[1:3, 1:4]
green = image[:, :, 1]
flat = image.reshape(-1)
chw = image.transpose(2, 0, 1)
shifted = image.astype(np.float32) + np.array([10, 20, 30], dtype=np.float32)

assert crop.shape == (2, 3, 3)
assert green.shape == (3, 4)
assert flat.size == image.size
assert chw.shape == (3, 3, 4)
assert np.array_equal(shifted[0, 0], [10, 21, 32])

print("image shape:", image.shape)
print("crop shape:", crop.shape)
print("green channel shape:", green.shape)
print("CHW shape:", chw.shape)
print("first shifted pixel:", shifted[0, 0])
