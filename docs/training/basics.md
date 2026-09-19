# Training Basics

Training uses [Keras](https://keras.io/) on [TensorFlow](https://www.tensorflow.org/).
From a JupyterLab terminal or SSH session:

```bash
docker exec -it cvbench bash
tm new train
train data --epochs 20 --backbone efficientnet_b0
# Ctrl+B D to detach — training continues after you close the terminal
```

`DATA_DIR` accepts a full path (`data/my_dataset`) or a bare dataset name
(`my_dataset`), the same convention `runs`/`evaluate` use for run names: a
bare name is resolved under `data/`, and a literal path is used as-is.

!!! tip
    Use `tm new <name>` to start a [tmux](https://github.com/tmux/tmux/wiki) session before a long training run.
    Detach with `Ctrl+B D` and the run keeps going after you close the
    terminal or disconnect SSH.

## Key options

| Option | Description |
|---|---|
| `--output <dir>` | Experiment output directory (default: `experiments/<auto-name>/`) |
| `--from <dir>` | Load config from an existing experiment as baseline |
| `--backbone` | Backbone name (`efficientnet_b0`..`b5`, `resnet_18`, `resnet_50`; see [Keras Applications](https://keras.io/api/applications/)). Detection defaults to `resnet_18` unless passed explicitly |
| `--weights` | Backbone weight init: `imagenet` (pretrained on [ImageNet](https://www.image-net.org/), default) or `none` (random/scratch) |
| `--epochs N` | Number of training epochs (see the warning below) |
| `--lr` | Learning rate |
| `--batch-size` | Batch size |
| `--input-size` | Image input size in pixels |
| `--dropout` | Dropout rate |
| `--augmentation` | Path to an augmentation YAML file, or the name of a saved `aug` config — see [Augmentation](../data/augmentation.md) |
| `--resume <checkpoint>` | Path to a checkpoint file to resume training from |
| `--class-weight` | Class weighting: `null` \| `auto` \| `'{"cat": 1.0, "dog": 2.5}'` |
| `--fine-tune-from-layer N` | Unfreeze backbone from this layer index onward (`0`=frozen, `-1`=all layers) — see [Two-Phase Training](two-phase.md) |
| `--val-split` | Fraction of train set used for validation when no `val/` directory exists (default: `0.2`) |
| `--seed N` | Random seed for reproducibility (sets Python, NumPy, and TensorFlow seeds) |

!!! warning
    `--epochs N` means *end at* epoch N, not *run N more epochs*. This matters
    most when resuming or fine-tuning — see [Two-Phase Training](two-phase.md)
    for the full explanation.

For optimizer and loss function options, see
[Optimizer & Loss](optimizer-loss.md). For learning rate scheduling, see
[Learning Rate Scheduling](lr-scheduling.md).

Next: [Evaluating a Run](../evaluation/evaluate.md).
