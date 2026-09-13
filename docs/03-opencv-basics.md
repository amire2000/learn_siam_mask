# Stage 3: OpenCV basics

## Goal

By the end, you will be able to:

- Load and save an image with OpenCV.
- Resize an image to an exact width and height.
- Convert between OpenCV's BGR order and RGB order.
- Draw a bounding box around an object.
- Display an image on a local desktop.

## Install OpenCV

This project already includes OpenCV. On a fresh clone, install everything
with:

```bash
uv sync
```

To add OpenCV to another `uv` project:

```bash
uv add opencv-python
```

Then import it as `cv2`:

```python
import cv2
```

## Intuition

OpenCV is a toolbox for moving between image files and image tensors. It loads
an image into a NumPy array, lets you transform or annotate that array, then
writes it back to an image file.

```text
image file ── cv2.imread ──> NumPy array ── OpenCV operations ──> cv2.imwrite ──> image file
```

For tracking, OpenCV commonly handles the practical image work around a model:
read a video frame, make a crop, draw the predicted box, and show the result.

## Load and save

```python
image = cv2.imread("frame.png")
if image is None:
    raise FileNotFoundError("frame.png could not be loaded")

cv2.imwrite("copy.png", image)
```

`cv2.imread` returns a NumPy array with shape `(H, W, 3)` for a normal color
image. Check for `None`: a misspelled path otherwise causes a less helpful
error later.

## BGR versus RGB

OpenCV loads color images in **BGR** order. Most plotting libraries and many
deep-learning examples use **RGB**. Convert deliberately when crossing that
boundary:

```python
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
```

The shape stays `(H, W, 3)`. Only channel meaning changes.

## Resize and draw a bounding box

```python
resized = cv2.resize(image, (320, 240))  # (width, height), unlike array shape
cv2.rectangle(resized, (40, 30), (180, 160), (0, 255, 0), thickness=2)
```

`cv2.resize` takes `(width, height)`, while NumPy reports image shape as
`(height, width, channels)`. This reversal is a common source of mistakes.

`cv2.rectangle` uses `(x, y)` corners and a BGR color. `(0, 255, 0)` is green
in both BGR and RGB because only its middle channel is nonzero.

## Display on your desktop

```python
cv2.imshow("Tracking result", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

This needs a local graphical desktop. It will not work in a headless terminal,
remote server, or automated test. In those environments, use `cv2.imwrite` to
save the result instead.

## Runnable example

Run this from the repository root:

```bash
uv run python examples/stage3_opencv_basics.py
```

The example creates a small BGR image, saves it, loads it again, resizes it,
draws a green box, and saves the result as `output/stage3_annotated.png`.

## Hands-on exercise

1. Change the output size from `320×200` to `160×100`.
2. Move the rectangle ten pixels to the right.
3. Change the rectangle to red. Remember: OpenCV color values are BGR.
4. On a local desktop, add the `cv2.imshow` code above and display the image.

## Review quiz

<form class="quiz" data-answer="b" data-explanation="OpenCV's default color order is BGR: blue, green, then red.">
  <fieldset>
    <legend>1. What channel order does <code>cv2.imread</code> normally return?</legend>
    <label><input type="radio" name="q1" value="a"> RGB</label><br>
    <label><input type="radio" name="q1" value="b"> BGR</label><br>
    <label><input type="radio" name="q1" value="c"> Grayscale</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="c" data-explanation="The first tuple item is width and the second is height, unlike NumPy's HWC shape order.">
  <fieldset>
    <legend>2. What does <code>cv2.resize(image, (320, 240))</code> request?</legend>
    <label><input type="radio" name="q2" value="a"> Height 320, width 240</label><br>
    <label><input type="radio" name="q2" value="b"> 320 color channels and height 240</label><br>
    <label><input type="radio" name="q2" value="c"> Width 320, height 240</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="a" data-explanation="The conversion code swaps the channel interpretation while retaining three channels and the same height and width.">
  <fieldset>
    <legend>3. What changes after BGR-to-RGB conversion?</legend>
    <label><input type="radio" name="q3" value="a"> The meaning of the color channels</label><br>
    <label><input type="radio" name="q3" value="b"> The image height and width</label><br>
    <label><input type="radio" name="q3" value="c"> The number of pixels</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="c" data-explanation="An unreadable or missing file makes <code>cv2.imread</code> return <code>None</code>.">
  <fieldset>
    <legend>4. What should you check after loading an image?</legend>
    <label><input type="radio" name="q4" value="a"> That its dtype is always <code>float32</code></label><br>
    <label><input type="radio" name="q4" value="b"> That it has exactly four channels</label><br>
    <label><input type="radio" name="q4" value="c"> That it is not <code>None</code></label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="b" data-explanation="<code>cv2.imshow</code> needs a graphical desktop. Saving the image gives you a file that you can copy or open elsewhere.">
  <fieldset>
    <legend>5. You run code through SSH on a server with no desktop. How can you inspect a processed frame?</legend>
    <label><input type="radio" name="q5" value="a"> Show it with <code>cv2.imshow</code></label><br>
    <label><input type="radio" name="q5" value="b"> Save it with <code>cv2.imwrite</code>, then open the file</label><br>
    <label><input type="radio" name="q5" value="c"> Convert it to a Python list</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

## Prerequisites and next step

Prerequisite: [Stage 2: NumPy image operations](02-numpy-image-operations.md).

Next: [Stage 4: Preprocessing](04-preprocessing.md) combines these operations
into model preprocessing: resize, normalization, mean/std, and HWC → CHW.
