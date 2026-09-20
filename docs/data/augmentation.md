# Augmentation

An augmentation config is a single YAML file. Keep it in `workspace/`, edit it
in any editor, and pass its path to `train --augmentation` (applied live
during training) or `data upsample --augmentation` (materialized to disk).

## Generate a config

`data aug` writes a starting file; you edit it from there:

```bash
data aug --preset light                       # → workspace/augmentation.yaml
data aug --preset heavy -o workspace/heavy.yaml
data aug                                      # reference sheet (see below)
```

| Preset | Contents |
|---|---|
| `light` | Horizontal flip + tiny rotation. Safe for any dataset. |
| `standard` | Flip, rotation, brightness/contrast, blur. A good general starting point. |
| `heavy` | Standard plus zoom, fog and salt-and-pepper noise. Aggressive regularisation. |
| `reference` (default) | Every available transform, commented out, with all parameters. Uncomment what you want. |

`data aug` refuses to overwrite an existing file unless you pass `--force`.

Each transform in a generated file carries a short comment explaining what it
does and what its parameters mean, so you can tweak values without looking
anything up:

```yaml
transforms:
  # Randomly rotate the image.
  - name: keras_rotation
    prob: 0.7  # chance this fires per image
    factor: 0.1

  # Gaussian blur.
  - name: aug_blur
    prob: 0.3  # chance this fires per image
    radius: 1.0  # range 0.5–15.0
```

Then use it:

```bash
train data/my_data --augmentation workspace/augmentation.yaml --epochs 30
```

!!! note
    The file is not validated when it is written, so a typo surfaces the next
    time you run `train` or `data upsample` with it.

## The reference sheet

`--preset reference` (the default) writes a fully commented file listing every
available transform, plus the syntax for range sampling and mutually exclusive
groups. Open it and uncomment the transforms you want:

```bash
data aug --preset reference -o workspace/aug_ref.yaml
```

Any numeric parameter can be written as a two-element list, e.g.
`radius: [0.5, 2.0]`, to sample a fresh value per image. Wrap candidates in a
`one_of` block so that at most one fires per image; `weight` sets their
relative frequency.

## Upsampling a class folder

Use `data upsample` to materialise an augmented copy of a single class folder
on disk. This is useful for correcting class imbalance before training —
apply it only to the classes that need more samples (e.g. skip the `noise`
class if it is already well-represented).

```bash
# Upsample the 'dog' class from however many originals it has to 1500 images.
# The destination folder must be empty or non-existent.
data upsample data/my_data/train/dog data/my_data_aug/train/dog \
  --augmentation workspace/augmentation.yaml \
  --target 1500
```

**What it does:**

1. Copies every original image to `dst_dir` with a fresh 16-char random hex filename.
2. Randomly picks source images and augments them until `--target` is reached.
3. Uses MD5 hashing to detect exact duplicates; retries up to 10 times per sample before skipping.
4. If the source already has ≥ `--target` images, the command exits with a hint to use `data downsample` instead (not yet implemented).

| Option | Required | Description |
|---|---|---|
| `--augmentation FILE` | ✓ | Path to an augmentation YAML file (same format as `--augmentation` in `train`) |
| `--target N` | ✓ | Total number of images the destination folder should contain |

Next: reshape the rest of the dataset with [Preparing Real Datasets](prepare.md), or move on to [Training Basics](../training/basics.md).
