# Augmentation

Augmentation configs are named, saved artifacts managed under
`workspace/augmentations/` — the same way `runs` manages `experiments/`.
Once saved, a config is used by name with `train --augmentation` (applied
live during training) or `data upsample --augmentation` (materialized to
disk).

## Discover what's available

```bash
aug transforms                    # every building-block transform + default params
aug list                          # every config you've saved
```

## Generate a config

`aug generate` shows a checklist of every available transform, each with a
short description (space to toggle, arrow keys to move, enter to confirm),
pre-checked with a preset's transforms if `--preset` is given, all unchecked
otherwise:

```text
$ aug generate --preset standard
? Select transforms to include (space to toggle, enter to confirm):
 » ● keras_flip        Randomly flip the image.
   ● keras_rotation    Randomly rotate the image.
   ● keras_brightness  Randomly adjust brightness.
   ● keras_contrast    Randomly adjust contrast.
   ● aug_blur          Gaussian blur.
   ○ aug_fog           Add a fog/haze effect.
   ○ ...

Save as [standard]: my_config
  ✓ Saved → workspace/augmentations/my_config.yaml  (5 transform(s))
  Open it in an editor to fine-tune any value.
  Usage:  train data/ --augmentation my_config
```

The saved file is ready to fine-tune — each transform gets a compact comment
explaining what it does and what its parameters mean, so you can open it in
any editor (VS Code, vim, ...) and tweak values without looking anything up:

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

```bash
aug show my_config                # print its YAML, syntax-highlighted
aug edit my_config                # open it in $EDITOR to fine-tune a value
train data/ --augmentation my_config --epochs 30
```

!!! note
    `aug edit` opens the file in `$EDITOR`/`$VISUAL` (falling back to a
    platform default) and writes back whatever you save — no validation, so a
    typo just surfaces the next time you run `train`/`upsample` with it. GUI
    editors need a "wait" flag to work here, e.g. `EDITOR="code --wait"` for
    VS Code.

**`reference` preset** bypasses the wizard and saves a commented-out file
showing every available transform (including range-sampling and `one_of`
syntax) — open it in an editor and uncomment what you want:

```bash
aug generate --preset reference --name aug_ref
```

## Upsampling a class folder

Use `data upsample` to materialise an augmented copy of a single class folder
on disk. This is useful for correcting class imbalance before training —
apply it only to the classes that need more samples (e.g. skip the `noise`
class if it is already well-represented).

```bash
# Upsample the 'dog' class from however many originals it has to 1500 images.
# The destination folder must be empty or non-existent.
data upsample data/my_data/train/dog data/my_data_aug/train/dog \
  --augmentation my_config \
  --target 1500
```

**What it does:**

1. Copies every original image to `dst_dir` with a fresh 16-char random hex filename.
2. Randomly picks source images and augments them until `--target` is reached.
3. Uses MD5 hashing to detect exact duplicates; retries up to 10 times per sample before skipping.
4. If the source already has ≥ `--target` images, the command exits with a hint to use `data downsample` instead (not yet implemented).

| Option | Required | Description |
|---|---|---|
| `--augmentation NAME\|FILE` | ✓ | A saved `aug` config name, or a path to an augmentation YAML file (same format as `--augmentation` in `train`) |
| `--target N` | ✓ | Total number of images the destination folder should contain |

Next: reshape the rest of the dataset with [Preparing Real Datasets](prepare.md), or move on to [Training Basics](../training/basics.md).
