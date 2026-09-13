# Stage 1: Image representation

## Goal

By the end, you will be able to:

- Read an image shape such as `480×640×3`.
- Explain pixels, color channels, and `uint8` values.
- Convert image data from `0…255` integers to `0.0…1.0` floats.

## Intuition

A grayscale image is a grid: one number per pixel. An RGB color image stores
three numbers at each pixel: red, green, and blue intensity.

```text
One RGB pixel: [red, green, blue]

Image shape: (H, W, C)
             │  │  └─ channels: 3 for RGB
             │  └──── width: columns of pixels
             └─────── height: rows of pixels
```

For example, a `480×640` RGB image usually has shape `(480, 640, 3)`:
480 rows, 640 columns, and three values per pixel.

## Pixels, channels, and values

The usual image data from OpenCV is an unsigned 8-bit integer array:

| Property | Meaning |
| --- | --- |
| `dtype` | `uint8`: an integer from 0 through 255 |
| `0` | no intensity in that channel |
| `255` | maximum intensity in that channel |
| `(H, W, 3)` | color image with three channels |

Deep-learning models commonly use `float32` values instead. A simple
normalization maps `0…255` to `0.0…1.0` by dividing by `255`.

!!! note
    OpenCV normally loads color pixels as **BGR**, not RGB. The tensor shape
    is still `(H, W, 3)`; only the meaning of the three values changes.

## Minimum math

For an RGB image `I`, the value at row `y`, column `x`, and channel `c` is:

`I[y, x, c]`

where `c = 0, 1, 2`. For RGB, those channels mean red, green, and blue.

The number of stored values is `H × W × C`. A `480×640×3` `uint8` image
therefore stores `921,600` one-byte values, about 0.88 MiB.

## Runnable example

Run this from the repository root:

```bash
uv run python examples/stage1_image_representation.py
```

It creates a tiny `2×3` RGB image so every value is easy to inspect.

```python
import numpy as np

image = np.array(
    [
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[255, 255, 255], [0, 0, 0], [128, 128, 128]],
    ],
    dtype=np.uint8,
)

print(image.shape)       # (2, 3, 3): H, W, C
print(image[0, 0])       # [255, 0, 0]: top-left RGB pixel
print(image.dtype)       # uint8
print(image / 255.0)     # float values from 0.0 to 1.0
```

## Hands-on exercise

Before running the example, answer these:

1. What is the shape of an RGB image that is 100 pixels high and 200 pixels wide?
2. Which pixel does `image[1, 2]` select in the example?
3. What value range do you expect after `image.astype(np.float32) / 255.0`?

Then change the top-left pixel to yellow. In RGB, yellow is `[255, 255, 0]`.

## Review quiz

<form class="quiz" data-answer="b" data-explanation="H counts image rows, so it is the height in pixels.">
  <fieldset>
    <legend>1. What does the <code>H</code> in <code>(H, W, C)</code> mean?</legend>
    <label><input type="radio" name="q1" value="a"> Number of channels</label><br>
    <label><input type="radio" name="q1" value="b"> Image height in pixels</label><br>
    <label><input type="radio" name="q1" value="c"> Pixel value range</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="a" data-explanation="A grayscale pixel needs only one intensity value.">
  <fieldset>
    <legend>2. How many channels does a grayscale image usually have?</legend>
    <label><input type="radio" name="q2" value="a"> 1</label><br>
    <label><input type="radio" name="q2" value="b"> 2</label><br>
    <label><input type="radio" name="q2" value="c"> 3</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="c" data-explanation="An unsigned 8-bit integer stores whole numbers from 0 through 255.">
  <fieldset>
    <legend>3. What is the valid range of a <code>uint8</code> pixel value?</legend>
    <label><input type="radio" name="q3" value="a"> -1 to 1</label><br>
    <label><input type="radio" name="q3" value="b"> 0.0 to 1.0</label><br>
    <label><input type="radio" name="q3" value="c"> 0 to 255</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="b" data-explanation="Continuous, consistently scaled float values are convenient for neural-network operations and training.">
  <fieldset>
    <legend>4. Why do models often use normalized <code>float32</code> images?</legend>
    <label><input type="radio" name="q4" value="a"> Floats need less memory than `uint8`.</label><br>
    <label><input type="radio" name="q4" value="b"> Training and model operations work naturally with continuous values.</label><br>
    <label><input type="radio" name="q4" value="c"> RGB images cannot be stored as `uint8`.</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="a" data-explanation="Both formats contain three channels; only which color each position represents changes.">
  <fieldset>
    <legend>5. What changes when an image is interpreted as BGR rather than RGB?</legend>
    <label><input type="radio" name="q5" value="a"> Channel meaning and order change; the shape stays `(H, W, 3)`.</label><br>
    <label><input type="radio" name="q5" value="b"> The height and width are swapped.</label><br>
    <label><input type="radio" name="q5" value="c"> The image becomes grayscale.</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

## Prerequisites and next step

Prerequisites: Python and basic NumPy arrays.

Next: [Stage 2: NumPy image operations](02-numpy-image-operations.md), where
you will crop, slice, reshape, transpose, and broadcast image tensors.
