# Stage 6: Filters as convolution

## Goal

By the end, you will be able to:

- Explain that a filter is a kernel with a useful job.
- Describe why averaging nearby pixels makes an image look blurry.
- Describe why comparing nearby pixels finds an edge.
- Run blur, sharpen, and edge filters with the convolution you already built.

## One process, different questions

In Stage 5, you learned one process:

1. Place a small kernel on an image patch.
2. Multiply matching numbers.
3. Add the results.
4. Slide the kernel and repeat.

The kernel numbers decide what question is being asked. A **filter** is simply
a kernel chosen to do a useful job.

```text
same sliding process + different kernel numbers = different filter
```

For this lesson, every image is grayscale. `0` means black, `1` means white,
and values between them are shades of gray.

## 1. Blur: average nearby pixels

This `3 × 3` blur kernel gives each nearby pixel the same small vote:

```text
1/9  1/9  1/9
1/9  1/9  1/9
1/9  1/9  1/9
```

All nine values add up to `1`. So the result is the average brightness of a
small square.

Imagine one very bright pixel surrounded by dark pixels. The average is not
very bright, so the bright point gets softer. Its brightness also spreads a
little to nearby output positions. That is blur.

```text
bright dot                 after averaging

0 0 0                      small gray values nearby
0 1 0          →           instead of one sharp white value
0 0 0
```

## 2. Sharpen: keep the center, subtract its neighbors

This sharpening kernel gives the center a strong positive vote and its four
direct neighbors negative votes:

```text
 0  -1   0
-1   5  -1
 0  -1   0
```

In a flat area, every nearby value is similar. The positive `5` and negative
values mostly balance, so the area stays similar. At a sudden change, the
center differs from its neighbors and the difference becomes stronger. That
makes boundaries look crisper.

Sharpening can create values below `0` or above `1`. To save a normal image,
you would clip them back into a valid range. For learning, it is useful to see
the raw numbers first.

## 3. Edge detection: look for a sudden change

This kernel compares the left side of a patch with its right side:

```text
-1   0   1
-1   0   1
-1   0   1
```

It is looking for a **vertical edge**: a boundary that goes up and down, such
as a black wall next to a white wall.

```text
black area | white area
0 0 0 1 1 1
        ↑
     vertical edge
```

On the dark left, the kernel gives negative values a negative vote and bright
right values a positive vote. Around the boundary, the result becomes large.
In an all-dark or all-bright patch, the negative and positive votes cancel to
zero, so there is no edge.

If you reverse the kernel signs, bright-to-dark edges produce positive output
instead. The important information is the large difference, not whether the
number is positive or negative.

## The important connection to deep learning

These filters are hand-written. A convolutional neural network starts with the
same slide/multiply/add operation, but it **learns** useful kernel numbers from
training examples. Early learned kernels often respond to edges, simple
textures, and color changes. Later layers combine those simple signals into
more useful features.

You do not need to memorize these three kernels. Learn the bigger idea:
**kernel numbers decide what pattern produces a large output.**

## Runnable example

Run this from the repository root:

```bash
uv run python examples/stage6_filters_as_convolution.py
```

It creates a black-left/white-right image and prints the blur, sharpen, and
vertical-edge outputs. The edge-filter output becomes large only near the
boundary.

## Hands-on exercise

1. Before running the script, predict the edge-filter output inside an area
   that is entirely black. Why?
2. Change the right half of the image from `1.0` to `0.5`. Does the edge
   response become stronger or weaker?
3. Reverse every sign in the edge kernel. What changes: the edge location, the
   output sign, or both?
4. Why does the blur-kernel sum need to be `1` if you want average brightness?

## Review quiz

<form class="quiz" data-answer="b" data-explanation="A filter is a kernel whose numbers are chosen to react to a useful local pattern, such as a blur or edge.">
  <fieldset>
    <legend>1. What makes a filter different from the generic kernel in Stage 5?</legend>
    <label><input type="radio" name="q1" value="a"> A filter never slides across an image.</label><br>
    <label><input type="radio" name="q1" value="b"> Its numbers are chosen to do a useful job.</label><br>
    <label><input type="radio" name="q1" value="c"> A filter can only be <code>1 × 1</code>.</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="a" data-explanation="Each <code>1/9</code> is one equal share. Nine shares add to 1, so the result is an average rather than a brighter or darker total.">
  <fieldset>
    <legend>2. Why do the values in the blur kernel add up to <code>1</code>?</legend>
    <label><input type="radio" name="q2" value="a"> So it computes an average brightness.</label><br>
    <label><input type="radio" name="q2" value="b"> So it creates more image channels.</label><br>
    <label><input type="radio" name="q2" value="c"> So it only finds edges.</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="c" data-explanation="In a flat patch, the left and right values are the same. Their negative and positive votes cancel, giving an output near zero.">
  <fieldset>
    <legend>3. What does the left-versus-right edge kernel output in a completely flat area?</legend>
    <label><input type="radio" name="q3" value="a"> A very large positive number</label><br>
    <label><input type="radio" name="q3" value="b"> A very large negative number</label><br>
    <label><input type="radio" name="q3" value="c"> A value near zero</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="b" data-explanation="The edge kernel compares left and right columns, so it responds at a boundary that runs vertically through the image.">
  <fieldset>
    <legend>4. What does the shown edge kernel look for?</legend>
    <label><input type="radio" name="q4" value="a"> A horizontal boundary, like floor below a wall</label><br>
    <label><input type="radio" name="q4" value="b"> A vertical boundary, like dark on the left and bright on the right</label><br>
    <label><input type="radio" name="q4" value="c"> A completely blurred image</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

<form class="quiz" data-answer="a" data-explanation="CNNs keep the same convolution operation, but training chooses kernel values that help solve the task.">
  <fieldset>
    <legend>5. What does a CNN learn about its filters during training?</legend>
    <label><input type="radio" name="q5" value="a"> Useful kernel numbers</label><br>
    <label><input type="radio" name="q5" value="b"> A new definition of image height</label><br>
    <label><input type="radio" name="q5" value="c"> How to remove every pixel</label>
  </fieldset>
  <button type="button" class="quiz-check">Check answer</button>
  <p class="quiz-result" aria-live="polite"></p>
</form>

## Prerequisites and next step

Prerequisite: [Stage 5: Convolution manually](05-convolution-manually.md).

Next: Stage 7 applies convolution to all three color channels and produces
several feature maps.
