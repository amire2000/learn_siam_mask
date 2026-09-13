# Stage 5: Convolution manually

## Goal

By the end, you will be able to:

- Describe convolution as a small grid of numbers sliding over a larger grid.
- Calculate one convolution output value by hand.
- Explain why an output can be smaller than its input.
- Write a small 2D convolution with NumPy and loops.

## Start with a picture made of numbers

For now, forget color images. Imagine a grayscale image: each pixel is one
number. Bigger numbers mean brighter pixels.

```text
image (4 × 4)

1   2   3   4
5   6   7   8
9  10  11  12
13 14  15  16
```

A **kernel** (also called a filter) is a much smaller grid of numbers. Here is
a `2 × 2` kernel:

```text
1  1
1  1
```

Convolution puts the kernel over a small part of the image, multiplies the
numbers in matching places, and adds the results. Then it slides the kernel
one pixel to the right and does it again.

Think of it like placing a small transparent stencil over a photo. The stencil
asks one tiny question at every position: “How much does this patch look like
the pattern I care about?”

## Calculate one output value by hand

Start with the kernel in the top-left corner:

```text
image patch          kernel             multiply, then add

1  2                 1  1              1×1 + 2×1 + 5×1 + 6×1
5  6                 1  1              = 14
```

So the first output value is `14`.

Slide the kernel one pixel right:

```text
2  3                 1  1              2×1 + 3×1 + 6×1 + 7×1
6  7                 1  1              = 18
```

Keep sliding right, then move down to the next row. The complete output is:

```text
14  18  22
30  34  38
46  50  54
```

## Why did the output get smaller?

The image is `4 × 4`, but the `2 × 2` kernel cannot start at the last row or
last column: it would hang off the edge. It has only three valid horizontal
positions and three valid vertical positions.

```text
input size:   4 × 4
kernel size:  2 × 2
output size:  3 × 3
```

With no padding and a one-pixel step, the rule is:

`output height = input height - kernel height + 1`

`output width = input width - kernel width + 1`

We will learn padding and bigger steps (stride) later. For now, the kernel
only visits places where it completely fits.

## What does this kernel mean?

The all-ones kernel adds the brightness of every `2 × 2` patch. Bright patches
produce large outputs; dark patches produce small outputs.

Different kernel numbers ask different questions. Later, you will use kernels
that react to edges and learn how a neural network learns useful kernel values
from examples. The sliding, multiply, and add process stays the same.

!!! note
    Deep-learning libraries usually call this operation “convolution,” even
    though they commonly skip a mathematical kernel flip. For this course, the
    practical CNN meaning is what matters: slide, multiply matching values,
    and add them.

## Runnable example

Run this from the repository root:

```bash
uv run python examples/stage5_convolution_manually.py
```

The script uses the same grid and kernel above. Its assertions check the full
`3 × 3` answer, so changing the convolution logic incorrectly makes it fail.

## Read the code slowly

```python
for y in range(output_height):
    for x in range(output_width):
        patch = image[y:y + kernel_height, x:x + kernel_width]
        output[y, x] = np.sum(patch * kernel)
```

- The two loops choose where the top-left corner of the kernel goes.
- `patch` cuts out the part of the image under that kernel.
- `patch * kernel` multiplies matching cells.
- `np.sum(...)` adds those multiplied cells into one output number.

Real deep-learning frameworks perform this much faster than Python loops, but
they do the same calculation.

## Hands-on exercise

1. Without running the code, calculate the top-middle output value. It uses
   the patch `[[2, 3], [6, 7]]`.
2. Change the kernel to `[[1, 0], [0, 1]]`. What is its first output value?
3. Use a `3 × 3` all-ones kernel on the `4 × 4` image. Predict the output
   shape before you run it.
4. Change one image value from `1` to `100`. Which output positions change?

## Review quiz

<form class="quiz" data-answer="b" data-explanation="At every valid position, a kernel multiplies matching cells in the image patch and then adds those products.">
  <fieldset>
    <legend>1. What is the main operation performed at one kernel position?</legend>
    <label><input type="radio" name="q1" value="a"> Sort every pixel in the image</label><br>
    <label><input type="radio" name="q1" value="b"> Multiply matching cells and add the results</label><br>
    <label><input type="radio" name="q1" value="c"> Replace every pixel with the kernel size</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="c" data-explanation="The top-left patch is <code>[[1, 2], [5, 6]]</code>. Adding those four values gives 14.">
  <fieldset>
    <legend>2. What is the first output value for the all-ones <code>2 × 2</code> kernel?</legend>
    <label><input type="radio" name="q2" value="a"> 6</label><br>
    <label><input type="radio" name="q2" value="b"> 10</label><br>
    <label><input type="radio" name="q2" value="c"> 14</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="a" data-explanation="A 2 × 2 kernel has three positions across and three down when it must fully fit inside a 4 × 4 image.">
  <fieldset>
    <legend>3. What is the output shape for a <code>4 × 4</code> image and <code>2 × 2</code> kernel, with no padding?</legend>
    <label><input type="radio" name="q3" value="a"> <code>3 × 3</code></label><br>
    <label><input type="radio" name="q3" value="b"> <code>4 × 4</code></label><br>
    <label><input type="radio" name="q3" value="c"> <code>2 × 2</code></label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="b" data-explanation="The kernel is the small pattern that moves over the larger image.">
  <fieldset>
    <legend>4. Which grid slides across the image?</legend>
    <label><input type="radio" name="q4" value="a"> The output grid</label><br>
    <label><input type="radio" name="q4" value="b"> The kernel</label><br>
    <label><input type="radio" name="q4" value="c"> The color-channel list</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="c" data-explanation="The 3 × 3 kernel fits in two positions across and two positions down: <code>4 - 3 + 1 = 2</code>.">
  <fieldset>
    <legend>5. What is the output shape for a <code>4 × 4</code> image and <code>3 × 3</code> kernel, with no padding?</legend>
    <label><input type="radio" name="q5" value="a"> <code>4 × 4</code></label><br>
    <label><input type="radio" name="q5" value="b"> <code>3 × 3</code></label><br>
    <label><input type="radio" name="q5" value="c"> <code>2 × 2</code></label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

## Prerequisites and next step

Prerequisites: [Stage 1: Image representation](01-image-representation.md)
and [Stage 2: NumPy image operations](02-numpy-image-operations.md).

Next: Stage 6 gives kernels useful jobs: blur, sharpen, and edge detection.
