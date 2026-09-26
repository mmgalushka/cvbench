# Preparing Real Datasets

The `data` command group has seven subcommands in total: `list` and `explore`
for inspecting a dataset, `upsample` for correcting class imbalance (see
[Augmentation](augmentation.md#upsampling-a-class-folder)), and four verbs for
reshaping a dataset (classification or
[YOLO](https://docs.ultralytics.com/datasets/detect/) layout) without touching the
source — clean, prep, split, and flatten. Each of the four copies
`SRC` to `DST` and leaves `SRC` untouched.

## Listing and inspecting datasets

```bash
data list                          # every dataset under data/, with task, split sizes, and class count
data explore data/my_data          # per-class brightness and class-balance report for the train split
data explore data/my_data --split test
```

`data list` labels each dataset classification or detection (auto-detected
from its layout — see [Your First Model](../getting-started/first-model.md))
and shows per-split image counts and the number of classes. `data explore`
reports mean brightness and image counts per class, useful for spotting
lighting bias or class imbalance before you train.

| Option | Description |
|---|---|
| `data explore --split` | Dataset split to analyse: `train` (default) / `val` / `test` |

### The four dataset-copy verbs

| Command | What it does |
|---|---|
| `data clean` | Drop OS/editor junk files |
| `data prep` | Rename images to content-hash filenames and drop exact duplicates |
| `data split` | Split a flat pool into train/val/test, stratified |
| `data flatten` | Pool an already-split dataset back into one flat folder |

All four share the same shape: `data <verb> SRC DST [options]`, and most
support `--dry-run` to preview the result before writing anything.

## Cleaning a dataset

Use `data clean` to copy a dataset (classification or YOLO layout) while
dropping OS/editor junk: `.DS_Store`, `Thumbs.db`, `__MACOSX/`,
`.Spotlight-V100`, AppleDouble shadow files (`._*`), and editor swap/temp
files. Directories left empty by junk removal are simply not created at the
destination. The source is never modified.

```bash
data clean data/my_data data/my_data_clean --dry-run   # preview
data clean data/my_data data/my_data_clean             # write the cleaned copy
```

| Option | Required | Description |
|---|---|---|
| `--dry-run` |  | Print what would be removed without writing `DST` |

## Prepping a dataset

Use `data prep` to copy a dataset (classification or YOLO layout) while
renaming every image to a content-hash filename (e.g.
`0ca9c69d9741cb49e9800998ecf8427e.png`) and dropping exact duplicates, in one
pass. The name is the full 32-hex-char MD5 digest of the image's pixel
content — deterministic and idempotent, the same image always gets the same
name in any dataset. At that length, two *different* images sharing a digest
is negligible, so a name collision reliably means duplicate content: within
each duplicate group only the lexicographically-first original path is kept
and copied; the rest are dropped. For YOLO, dropping an image also drops its
paired label file. `--across-splits` additionally flags duplicate groups
whose members span more than one split (train/val/test) — the highest-value
check, since that's data leakage between splits.

```bash
data prep data/my_data data/my_data_prepped --across-splits
```

| Option | Required | Description |
|---|---|---|
| `--across-splits` |  | Warn when a duplicate group spans more than one split |
| `--dry-run` |  | Print the plan without writing `DST` |

!!! warning
    Run `data prep` (and especially `--across-splits`) before `data split`
    if you're not sure the source is clean — duplicates that leak across
    train/val/test inflate validation and test metrics.

## Splitting a dataset

Use `data split` to copy a dataset (classification or YOLO layout) into
train/val/test,
[stratified](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
by class (every split keeps the dataset's class proportions). `SRC` must be a flat pool
(classification: `<class>/*`; YOLO: `images/*` + `labels/*`) — an
already-split `SRC` is rejected; run `data flatten` first, then re-split the
result. YOLO images can carry boxes of more than one class, so the
stratification key is each image's *primary* (most frequent, ties broken by
lowest id) box class; images with no boxes are split the same proportional,
seeded way as every other group.

```bash
data split data/my_data data/my_data_split --train 0.8 --val 0.1 --test 0.1 --seed 42
```

| Option | Required | Description |
|---|---|---|
| `--train FLOAT` |  | Fraction assigned to train (default: 0.8) |
| `--val FLOAT` |  | Fraction assigned to val (default: 0.1) |
| `--test FLOAT` |  | Fraction assigned to test (default: 0.1) |
| `--seed N` |  | Random seed for the stratified shuffle (default: 42) |
| `--dry-run` |  | Print the planned split without writing `DST` |

## Flattening a dataset

Use `data flatten` to copy an already-split dataset (classification or YOLO
layout) back into one flat pool — the exact inverse of `data split`. Every
image from every split (`train`/`val`/`test`) is copied into a single flat
destination (classification: `<class>/*`; YOLO: `images/*` + `labels/*`), with
no split structure left. A `SRC` that isn't already split is rejected —
there's nothing to flatten. This is the way to re-split a dataset with
different ratios or a different seed: flatten it, then split the flattened
result.

```bash
data flatten data/my_data_split data/my_data_flat
data split data/my_data_flat data/my_data_resplit --train 0.7 --val 0.15 --test 0.15
```

| Option | Required | Description |
|---|---|---|
| `--dry-run` |  | Print the flatten plan without writing `DST` |

Next: correct class imbalance with [Augmentation](augmentation.md#upsampling-a-class-folder), or move on to [Training Basics](../training/basics.md).
