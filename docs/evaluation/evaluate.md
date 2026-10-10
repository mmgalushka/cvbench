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
| `--conf <0-1>` | Detection runs only: score threshold for TP/FP/FN counts and precision/recall (default: the run's `detection.conf_threshold`). AP/mAP always use all scores |

```bash
evaluate det_yolo_2026_01_21 --conf 0.4
```

Evaluation writes `eval_report.json` into the run directory (or
`--output-dir`), which both `runs show <run>` and the WebUI's Evaluation tab
read to render results.

## What gets reported

The report shape depends on the run's task:

- **Classification** — overall accuracy, per-class
  [precision/recall/F1](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures), and a
  [confusion matrix](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) with clickable-in-the-WebUI misclassified samples.
  The validation accuracy is shown beside the test accuracy, and top-3 accuracy
  appears only for datasets with more than 5 classes.
- **Detection** — a per-class outcome table (matched / mislocated / confused
  / missed) plus AP/precision/recall/F1 and a class-confusion matrix. Ground
  truth uses the [YOLO txt format](https://docs.ultralytics.com/datasets/detect/).
  Every TP/FP/FN count is labelled with the conf and IoU thresholds it was
  computed at.

Evaluation also prints advisories when a run looks undertrained: *many false
positives per image* (detection, 2 or more per image) and *validation loss was
still falling when training stopped* (train longer).

!!! tip "Reading detection metrics"
    A predicted box counts as a match when its overlap with a ground-truth box
    (IoU, intersection over union) passes a threshold. AP (average precision)
    summarizes the precision/recall trade-off across confidence thresholds for
    one class.

See [Experiment Tracker](../tools/experiment-tracker.md) for how the WebUI
presents these results, and `runs show <run>` for the terminal equivalent.

Next: [Running Predictions](predict.md), or package the run for a device with
[Export Formats](../deployment/export.md).
