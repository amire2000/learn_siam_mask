# Roadmap: from images to SiamMask

## Final goal

Understand SiamMask as an object tracker that predicts both a bounding box and
a pixel-level mask. You should be able to trace the shape and purpose of every
major tensor during inference.

## Big milestones

| Milestone | Stages | Outcome |
| --- | --- | --- |
| Images in memory | 1–4 | Prepare an image tensor correctly for a model. |
| Convolutional features | 5–9 | Explain how filters create spatial feature maps. |
| Efficient backbones | 10–15 | Build and reason about a small feature extractor. |
| Siamese tracking | 16–20 | Locate a target by matching template and search features. |
| Box tracking in practice | 21–25 | Trace a NanoTrack-style model and run it through ONNX. |
| SiamMask | 26–32 | Explain and trace the extra mask-prediction branch. |

## Course path

| Stage | Topic | What you should be able to do |
| --- | --- | --- |
| 1 | [Image representation](01-image-representation.md) | Understand `H×W×C`, pixels, channels, dtype, and ranges. |
| 2 | [NumPy image operations](02-numpy-image-operations.md) | Slice, crop, reshape, transpose, and broadcast. |
| 3 | OpenCV basics | Load, display, resize, convert color, and draw a bounding box. |
| 4 | Preprocessing | Resize, normalize, use mean/std, and convert HWC → CHW. |
| 5 | Convolution manually | Implement a small 2D convolution with NumPy. |
| 6 | Filters as convolution | Use blur, sharpen, and edge-detection filters. |
| 7 | Multi-channel convolution | Turn RGB input and several kernels into feature maps. |
| 8 | Stride and padding | Explain why feature maps shrink or stay the same size. |
| 9 | Feature maps | Visualize what simple filters detect. |
| 10 | Activation functions | Explain ReLU and why nonlinearities matter. |
| 11 | Pooling/downsampling | Use max/average pooling and explain spatial reduction. |
| 12 | `1×1` convolution | Explain channel mixing. |
| 13 | Depthwise convolution | Apply one spatial filter per channel. |
| 14 | Depthwise-separable convolution | Combine depthwise and pointwise convolution: the MobileNet idea. |
| 15 | Backbone concept | Build a tiny feature extractor from these blocks. |
| 16 | Template/search crops | Extract target and search regions for a tracker. |
| 17 | Feature matching | Compare two feature tensors. |
| 18 | Cross-correlation | Build a response map from template and search features. |
| 19 | Find target location | Use a response-map maximum to estimate motion. |
| 20 | Simple Siamese tracker | Track a synthetic object using handcrafted features. |
| 21 | Classification output | Explain target/background score maps. |
| 22 | Box regression | Explain predicted bounding-box distances or coordinates. |
| 23 | NanoTrack tensor flow | Trace backbone and head inputs and outputs. |
| 24 | ONNX inference | Feed NanoTrack with OpenCV or ONNX Runtime. |
| 25 | Replace pieces mentally | Connect each NanoTrack operation to what you built. |
| 26 | Binary segmentation masks | Represent foreground and background as an `H×W` mask. |
| 27 | Mask labels and resizing | Align masks with image crops and output resolution. |
| 28 | Segmentation losses | Understand binary cross-entropy and IoU practically. |
| 29 | SiamMask architecture | Identify its backbone, tracking head, and mask head. |
| 30 | SiamMask mask prediction | Trace target location to its predicted object mask. |
| 31 | SiamMask inference | Follow template, search crop, box prediction, and mask refinement. |
| 32 | NanoTrack vs SiamMask | Explain the accuracy/compute trade-off of predicting shape. |

## How each lesson works

Each lesson has intuition first, only the math it needs, a small runnable
example, an exercise, review questions, prerequisites, and the next step.
