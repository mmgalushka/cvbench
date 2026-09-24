# Hyperparameter Sweeps

`sweep` automates the "try a few settings and compare" loop. Give it
comma-separated values for any `train` flag and it trains **every combination**
(a grid search), then reports the best trial.

```bash
sweep data/shapes \
  --lr 1e-3,1e-4 \
  --backbone efficientnet_b0,resnet_18 \
  --epochs 20 \
  --name shapes_lr_backbone
```

Two learning rates times two backbones gives **4 trials**. `--epochs 20` has a
single value, so it is fixed for all of them.

## How flags map

Every sweepable flag takes a comma-separated list:

- **One value** — fixed for all trials (same as passing it to `train`).
- **Two or more values** — an *axis*. The grid is the product of all axes.

Sweepable flags: `--backbone`, `--weights`, `--epochs`, `--lr`, `--batch-size`,
`--input-size`, `--dropout`, `--augmentation`, `--class-weight`, `--loss`,
`--optimizer`, `--lr-scheduler`, `--fine-tune-from-layer`, `--val-split`,
`--seed`.

`--optimizer`, `--loss` and `--lr-scheduler` use the same `type:params` syntax
as `train` (see [Optimizer & Loss](optimizer-loss.md) and
[Learning Rate Scheduling](lr-scheduling.md)):

```bash
sweep data/shapes --optimizer adam,sgd:momentum=0.9 --lr 1e-3,1e-4
```

Flags you do not pass use the same defaults as `train`. To start from an
existing run instead, use `--from`:

```bash
sweep data/shapes --from experiments/baseline --lr 1e-3,3e-4,1e-4
```

Other options:

| Option | Description |
|---|---|
| `--name` | Sweep name; also the directory under `experiments/` and the trial prefix |
| `--strategy` | Search strategy. Only `grid` is available for now |
| `--metric` | Selection metric: `val_loss`, `val_accuracy` or `map50` |
| `--from EXP` | Use an existing experiment's config as the baseline |
| `--show` | Print the resolved trials and their count, without training |

The default metric is `val_loss` (lower is better) for classification and
`map50` (higher is better) for detection.

!!! note "No numeric ranges yet"
    Values must be listed explicitly. `:` ranges such as `0.1:0.5` are rejected;
    random search, which will use them, is a planned follow-up.

## Check the grid first

The number of trials multiplies with every axis: three axes of three values is
27 full training runs. Always preview with `--show`:

```bash
sweep data/shapes --lr 1e-3,1e-4 --backbone efficientnet_b0,resnet_18 --epochs 20 --show
```

!!! warning "Cost"
    Trials run one after another, each a complete training run. Detection
    trials can take hours each. Keep grids small, and use fewer `--epochs` to
    screen candidates before training the winner for longer.

## Output layout

A sweep is a directory of ordinary experiments plus a small manifest:

```text
experiments/
└── shapes_lr_backbone/          the sweep directory
    ├── sweep.yaml               manifest
    ├── shapes_lr_backbone_001/  trial: a normal experiment directory
    ├── shapes_lr_backbone_002/  (config.yaml, training_log.csv, ...)
    └── ...
```

`sweep.yaml` records what you asked for: name, date, data directory, strategy,
metric, direction (min or max) and the axes. It does not duplicate results;
those are read live from each trial's `config.yaml`, so a trial's status and
metrics are always current.

## Reading the results

When the sweep finishes it prints a summary table with one row per trial (the
axis values, the metric and the status) and names the best trial. Trials are
compared on the **validation split**, so the test split plays no part in
choosing the winner. The sweep never evaluates on the test split itself; that is
a separate, manual step. Score the best trial once you have decided on it:

```bash
evaluate shapes_lr_001
```

or get a test score for **every** trial by evaluating the sweep by name:

```bash
evaluate shapes_lr
```

Each finished trial is evaluated and gets its own `eval_report.json`, and a table
of the test scores is printed at the end. This is for information only: the best
trial was picked on validation and does not change. It costs one evaluation pass
per trial, so run it when you want the comparison.

- A trial that fails is recorded as failed and the sweep **continues** with the
  next one.
- A planned trial whose directory no longer exists is shown as `missing`.

Trial directories are regular experiments, so you can evaluate or predict with
any of them as usual:

```bash
evaluate shapes_lr_backbone_003
```

Trial names are unique, so a bare trial name works everywhere a run name does
(`runs show`, `evaluate`, `predict`, `runs export`, ...).

## In `runs list`

A sweep appears in `runs list` as **one row**, tagged `cls·sweep` or
`det·sweep` in the Task column. Val Loss and Epochs come from the best trial,
and Status is `running` while any trial is still training. To see the trials,
list the sweep directory:

```bash
runs list shapes_lr_backbone
```

How each `runs` command treats a sweep name:

| Command | On a sweep |
|---|---|
| `runs list` | one summary row; `runs list <sweep>` lists its trials |
| `runs show <sweep>` | the sweep's settings and trial table |
| `runs delete <sweep>` | removes the whole sweep and its trials |
| `runs rename`, `runs compare`, `runs export` | not supported: a clear message asks for a trial name |

`evaluate` and `predict` likewise ask for a trial name. Trials themselves are
ordinary runs for every command, except that `runs rename` refuses them
(renaming would make the sweep report the trial as missing).

## In the WebUI

The **Experiments** page lists a sweep as one row (tagged `sweep`) alongside
ordinary runs, with the best trial's validation loss. Click it to open the sweep
page: its settings, the axes, and a trial table with the best trial highlighted.
Click a trial to open it as a normal run; its back link returns to the sweep.
A sweep can be deleted from its page (this removes all trials). Sweeps and
trials cannot be renamed, and the Inference and Compare pickers do not list
trials yet.

## Limits

- **Grid only.** Random search, numeric ranges and presets are planned.
- **Sequential.** Trials do not run in parallel.
- **No resume.** An interrupted sweep is not continued; start a new one with a
  new `--name`. A sweep name that is already in use is rejected.
