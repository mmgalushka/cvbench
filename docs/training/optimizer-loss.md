# Optimizer & Loss

Both the optimizer and loss function are single-flag `train` options with the
same config shape: a type name, optionally followed by `:key=value,...`
parameters. Both are saved to `config.yaml` and applied automatically when
resuming a run.

## Optimizer

By default training uses Adam. Use `--optimizer` to switch to SGD or to add
weight decay (L2 regularization).

```bash
# Adam with weight decay
train data/ --optimizer adam:weight_decay=1e-4

# SGD with momentum and weight decay
train data/ --optimizer sgd:weight_decay=1e-4,momentum=0.9
```

| Option | Default | Description |
|---|---|---|
| `--optimizer adam` | ✓ | Adam optimizer |
| `--optimizer sgd` | — | SGD optimizer |
| `weight_decay=F` | `0.0` | L2 regularization penalty |
| `momentum=F` | `0.9` | Momentum (SGD only) |

## Loss function

By default training uses categorical cross-entropy. Use `--loss` to switch to
focal loss or to enable label smoothing, which are particularly useful when
you have false positive problems.

**Focal loss** shifts gradient weight toward hard, misclassified examples and
away from easy ones — forcing the model to focus on the ambiguous
signal/noise boundary:

```bash
# Focal loss with default gamma (2.0)
train data/ --loss focal

# Tune the focusing parameter
train data/ --loss focal:gamma=1.5
```

**Label smoothing** prevents the model from becoming overconfident, making
threshold-based rejection more reliable:

```bash
train data/ --loss crossentropy:label_smoothing=0.1
```

**Both combined** — focal loss with smoothing:

```bash
train data/ --loss focal:gamma=2.0,label_smoothing=0.1
```

| Option | Default | Description |
|---|---|---|
| `--loss crossentropy` | ✓ | Standard categorical cross-entropy |
| `--loss focal` | — | Focal loss (gamma=2.0) — focuses on hard examples |
| `gamma=F` | `2.0` | Focusing parameter; higher = stronger focus on hard examples |
| `label_smoothing=F` | `0.0` | Smooths targets; applies to both loss types |

!!! note
    Both configs are saved to `config.yaml` and applied automatically when
    resuming a run — you don't need to repeat `--optimizer`/`--loss` on a
    `--resume` call.

Next: [Learning Rate Scheduling](lr-scheduling.md).
