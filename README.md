# Learn SiamMask

A beginner-friendly learning site that builds from image tensors and
convolution to Siamese object tracking and SiamMask.

## Install

This project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github-amire:amire2000/siam_mask.git
cd siam_mask
uv sync
```

## Run the site

```bash
uv run mkdocs serve
```

Open <http://127.0.0.1:8000/> in a browser. The server reloads when you edit
a page.

## Check the site

```bash
uv run mkdocs build --strict
```

## Run a lesson example

```bash
uv run python examples/stage1_image_representation.py
```

## Layout

- `docs/` — MkDocs lessons and site assets
- `examples/` — small runnable Python examples
- `mkdocs.yml` — site configuration and navigation
- `docs/roadmap.md` — the full learning path
