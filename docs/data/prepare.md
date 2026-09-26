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
| `data prep` | Drop corrupt images and cross-split duplicates; hash-rename and dedup |
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
removing images that would hurt training or inflate evaluation. In one pass it:

- **Drops corrupt images** — empty or undecodable files are skipped and listed
  with the reason instead of crashing the command.
- **Removes cross-split duplicates** — the same image in more than one of
  train/val/test is data leakage, so only the copy in the highest-priority
  split is kept (train > val > test). For YOLO the paired label file is
  dropped too, with a warning if the labels differ.
- **Fails on label conflicts** — the same image filed under different classes
  (classification) aborts the command and lists the conflicting paths.
- **Renames to content hashes and drops within-split duplicates** (default) —
  the name is the full 32-hex-char MD5 digest of the pixel content (e.g.
  `0ca9c69d9741cb49e9800998ecf8427e.png`), deterministic and idempotent. Of
  each duplicate group the lexicographically-first path is kept.

The summary lists every dropped file and shows per-split image counts before
and after.

```bash
data prep data/my_data data/my_data_prepped
data prep data/my_data data/my_data_prepped --no-hash --no-duplicates
```

| Option | Required | Description |
|---|---|---|
| `--no-hash` |  | Keep original filenames (images are still hashed internally to find duplicates); warns when within-split duplicates are present |
| `--no-duplicates` |  | Drop within-split duplicates; only needed together with `--no-hash` |
| `--dry-run` |  | Print the plan without writing `DST` |

!!! warning
    Run `data prep` before `data split` if you're not sure the source is
    clean — duplicates that leak across train/val/test inflate validation and
    test metrics.

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
