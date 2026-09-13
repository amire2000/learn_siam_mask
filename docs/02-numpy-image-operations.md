# Stage 2: NumPy image operations

## Goal

By the end, you will be able to:

- Crop an image and select a single color channel.
- Reshape an image without changing its number of values.
- Convert HWC image data to CHW.
- Apply one value per channel with broadcasting.

## Intuition

An image tensor is a NumPy array, so image operations are array operations.
The key convention is that images use **row first, then column**:

```text
image[y, x, channel]
      │  │      └─ 0, 1, or 2
      │  └──────── x: left to right
      └─────────── y: top to bottom
```

For a crop, select a range of rows and columns. The end of each NumPy slice
is excluded, just like `range`.

```python
crop = image[y_start:y_end, x_start:x_end]
```

This is the same basic operation a Siamese tracker later uses to make its
template and search crops.

## Five useful operations

### 1. Slice and crop

```python
top_left = image[0:2, 0:3]
```

For an HWC image, leaving out the channel axis keeps all channels. A crop from
rows `0, 1` and columns `0, 1, 2` has shape `(2, 3, 3)`.

### 2. Select one channel

```python
red = image[:, :, 0]
```

`:` means “all values on this axis.” This removes the channel axis, so `red`
has shape `(H, W)`.

### 3. Reshape

```python
flat = image.reshape(-1)
```

`reshape` changes how you organize the same values; it does not change their
order or count. `-1` asks NumPy to calculate that dimension.

### 4. Transpose HWC to CHW

```python
chw = image.transpose(2, 0, 1)
```

Images from OpenCV usually use HWC. Many deep-learning models expect CHW:
channels first, then height, then width.

### 5. Broadcast one value per channel

```python
offset = np.array([10, 20, 30])
shifted = image + offset
```

NumPy repeats the three-value `offset` across every `(y, x)` position. This is
called broadcasting. It is useful for per-channel normalization later.

!!! warning
    Adding to a `uint8` image can overflow: `250 + 10` does not produce 260.
    Convert to `float32` before arithmetic for model preprocessing.

## Minimum math

Cropping takes a rectangular subset:

`crop = I[y_0:y_1, x_0:x_1, :]`

Its shape is `(y_1 - y_0, x_1 - x_0, C)`. Transposing HWC to CHW changes
only axis order: `(H, W, C) → (C, H, W)`.

## Runnable example

Run this from the repository root:

```bash
uv run python examples/stage2_numpy_image_operations.py
```

The image values are numbered so you can see exactly which values move or are
selected.

## Hands-on exercise

Use the example image to do the following before looking at its output:

1. Make a crop containing the bottom two rows and left two columns.
2. Select the blue channel from the image.
3. Convert its HWC shape to CHW.
4. Convert to `float32`, then subtract `[10, 20, 30]` from every pixel.

## Review quiz

<form class="quiz" data-answer="b" data-explanation="The first image axis is the row axis, which is the vertical y coordinate.">
  <fieldset>
    <legend>1. In <code>image[y, x, c]</code>, what does <code>y</code> select?</legend>
    <label><input type="radio" name="q1" value="a"> The color channel</label><br>
    <label><input type="radio" name="q1" value="b"> The row, from top to bottom</label><br>
    <label><input type="radio" name="q1" value="c"> The column, from left to right</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="c" data-explanation="The slice end is excluded: rows 1 and 2, columns 2, 3, and 4 give height 2 and width 3.">
  <fieldset>
    <legend>2. What is the shape of <code>image[1:3, 2:5]</code> from an RGB image?</legend>
    <label><input type="radio" name="q2" value="a"> <code>(3, 2, 3)</code></label><br>
    <label><input type="radio" name="q2" value="b"> <code>(2, 3)</code></label><br>
    <label><input type="radio" name="q2" value="c"> <code>(2, 3, 3)</code></label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="a" data-explanation="`transpose(2, 0, 1)` moves the channel axis from last to first.">
  <fieldset>
    <legend>3. Which operation converts HWC image data to CHW?</legend>
    <label><input type="radio" name="q3" value="a"> <code>image.transpose(2, 0, 1)</code></label><br>
    <label><input type="radio" name="q3" value="b"> <code>image.reshape(3, -1)</code></label><br>
    <label><input type="radio" name="q3" value="c"> <code>image[:, :, 0]</code></label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="b" data-explanation="The vector has one value for each final-axis channel, so NumPy repeats it across height and width.">
  <fieldset>
    <legend>4. Why can an HWC RGB image add a shape <code>(3,)</code> vector?</legend>
    <label><input type="radio" name="q4" value="a"> NumPy first converts the image to grayscale.</label><br>
    <label><input type="radio" name="q4" value="b"> Broadcasting applies the three values to every pixel's channels.</label><br>
    <label><input type="radio" name="q4" value="c"> `uint8` arrays always have three dimensions.</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="c" data-explanation="A `uint8` value cannot represent negative values or values above 255, so use `float32` before preprocessing arithmetic.">
  <fieldset>
    <legend>5. Why convert <code>uint8</code> data to <code>float32</code> before normalization?</legend>
    <label><input type="radio" name="q5" value="a"> <code>uint8</code> has no image shape.</label><br>
    <label><input type="radio" name="q5" value="b"> Only <code>float32</code> can have three channels.</label><br>
    <label><input type="radio" name="q5" value="c"> It prevents integer overflow and supports fractional values.</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

## Prerequisites and next step

Prerequisite: [Stage 1: Image representation](01-image-representation.md).

Next: Stage 3 will use OpenCV to load, resize, convert, and draw on real
images.
