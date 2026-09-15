# Learning Repository

This repository is a personal technical learning site.

## Goal

Teach topics incrementally from fundamentals to implementation.

The current main learning track is:

image tensors → convolution → feature maps → MobileNet → Siamese network → cross-correlation → classification/regression head → NanoTrack code.

## Documentation rules

When adding a topic:

1. Explain the intuition first.
2. Add the minimum mathematics required.
3. Add diagrams when useful.
4. Add a small runnable example.
5. Add a hands-on exercise.
6. Add 3-5 review questions.
7. Link prerequisites.
8. Link the next topic.
9. Start with a short “By the end, you will be able to” list.

Do not introduce advanced concepts before their prerequisites.

## Reader background

The reader knows:
- Python
- NumPy basics
- OpenCV basics
- Modern C++
- basic YOLO inference

The reader is learning deep learning architecture.

## Teaching style

Explain concepts as if teaching a curious 15-year-old student:

- Start with plain-language intuition and concrete examples.
- Introduce only the mathematics needed for the current topic.
- Explain every symbol and connect each formula to the intuition.
- Prefer short steps over assumed deep-learning knowledge.

## Site

Documentation lives in `docs/`.

Use MkDocs Material.

After modifying documentation run:

    mkdocs build

## Learning roadmap

See:

    docs/roadmap.md

## quizzes

- 5 questions
- difficulty: beginner -> intermediate
- multiple-choice options
- a clear “Check answer” button
- show a green check or red cross and explain the answer immediately
