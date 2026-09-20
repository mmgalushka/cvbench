# Generating Synthetic Data

`data generate` writes a synthetic geometric-shapes dataset for pipeline
testing — useful as a smoke test before you point CVBench at a real dataset.

```bash
docker exec cvbench data generate --train 200 --val 50 --test 50
```

This writes to `data/synthetic/` by default. Point training at it immediately:

```bash
docker exec -it cvbench bash
train data/synthetic --epochs 5 --backbone efficientnet_b0
```

## Classification dataset

```bash
data generate data/synthetic --train 200 --val 50 --test 50
```

```text
data/synthetic/
├── train/
│   ├── circle/0000.jpg
│   ├── square/0000.jpg
│   ├── triangle/0000.jpg
│   └── star/0000.jpg
├── val/     (same 4 class folders)
└── test/    (same 4 class folders)
```

- One shape per image; the class is the folder name — no label files.
- `--train/--val/--test` count images **per class**.

## YOLO txt dataset

This is the [YOLO detection format](https://docs.ultralytics.com/datasets/detect/):
one image per label file, one `class_id x_center y_center width height` line
per box, all coordinates normalized to 0–1.

```bash
data generate data/synthetic_yolo --format yolo \
    --train 200 --val 50 --test 50 --image-size 160 --max-objects 4
```

```text
data/synthetic_yolo/
├── data.yaml                # class names + split paths
├── images/
│   ├── train/0000.jpg
│   ├── val/0000.jpg
│   └── test/0000.jpg
└── labels/
    ├── train/0000.txt       # one "class_id xc yc w h" line per shape, normalized
    ├── val/0000.txt
    └── test/0000.txt
```

- 1–`--max-objects` shapes per image (default 3), each with a bounding box.
- `--train/--val/--test` count images **per split**, since one image can hold
  several classes.
- Every image has a same-named `.txt` next to it in `labels/`.

!!! tip
    The WebUI **Datasets** page reads both layouts: it labels each dataset's
    format and, for YOLO, draws the bounding boxes over every thumbnail
    (*Show boxes* toggles the overlay, the class filter keeps only images
    containing a given class). See [Experiment Tracker](../tools/experiment-tracker.md).

Next: point `train` at either dataset — see [Training Basics](../training/basics.md).
