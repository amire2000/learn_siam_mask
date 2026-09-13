"""Practice loading, resizing, converting, and annotating an image with OpenCV."""

from pathlib import Path

import cv2
import numpy as np


output = Path("output")
output.mkdir(exist_ok=True)
source_path = output / "stage3_source.png"
annotated_path = output / "stage3_annotated.png"

source = np.zeros((100, 160, 3), dtype=np.uint8)
source[:, :] = (255, 0, 0)  # Blue in OpenCV's BGR order.
cv2.imwrite(str(source_path), source)

image = cv2.imread(str(source_path))
if image is None:
    raise FileNotFoundError(source_path)

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized = cv2.resize(image, (320, 200))
cv2.rectangle(resized, (60, 40), (260, 160), (0, 255, 0), thickness=3)
cv2.imwrite(str(annotated_path), resized)

assert image.shape == (100, 160, 3)
assert rgb[0, 0].tolist() == [0, 0, 255]
assert resized.shape == (200, 320, 3)
assert annotated_path.exists()

print("loaded shape:", image.shape)
print("resized shape:", resized.shape)
print("saved:", annotated_path)
