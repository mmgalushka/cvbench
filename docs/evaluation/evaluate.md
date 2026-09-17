# Evaluating a Run

`evaluate` scores a trained model on the held-out test split.

```bash
evaluate cls_effnet_b0_2026_01_21
```

`EXPERIMENT` is the run name (e.g. `effnet_b3_lr5e5_cutmix_trial_2024_01_21`)
or a full path to the run directory. If a bare name is given, it is resolved
under `experiments/`.

```bash
evaluate experiments/cls_effnet_b0_2026_01_21 --output-dir /tmp/eval
```

| Option | Description |
|---|---|
| `--output-dir <dir>` | Where to write eval outputs (default: run dir) |

Evaluation writes `eval_report.json` into the run directory (or
`--output-dir`), which both `runs show <run>` and the WebUI's Evaluation tab
read to render results.

## What gets reported

The report shape depends on the run's task:

- **Classification** — overall accuracy, per-class precision/recall/F1, and a
  confusion matrix with clickable-in-the-WebUI misclassified samples.
- **Detection** — a per-class outcome table (matched / mislocated / confused
  / missed) plus AP/precision/recall/F1 and a class-confusion matrix.

See [Interactive Confusion Matrix](../webui/confusion-matrix.md) for how the
WebUI presents these results, and `runs show <run>` for the terminal
equivalent.

Next: [Running Predictions](predict.md), or package the run for a device with
[Export Formats](../deployment/export.md).
