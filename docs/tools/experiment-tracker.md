# Experiment Tracker

Start the WebUI with:

```bash
serve --host 0.0.0.0 --port 8000
```

It starts automatically when the container starts (set its advertised URL
with the `CVBENCH_URL` environment variable) and is reachable at
`http://<server-ip>:8000`.

<!-- IMAGE PLACEHOLDER: WebUI runs list screenshot — see design spec for capture instructions. Suggested alt text: "CVBench WebUI runs table showing name, backbone, date, status, and validation accuracy for each experiment" -->

## Runs (home page)

A table of every experiment: name, backbone, date, status, and validation
accuracy. Sortable and filterable. Click a row to open its run detail page.

## Run detail

Each run has four tabs:

### Training

Live loss/accuracy curves during training (streamed as the run progresses),
static afterwards, plus a config summary: backbone, learning rate, epochs,
and augmentations.

### Evaluation

Overall accuracy and a per-class
[precision/recall/F1](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures)
table, plus an interactive
[confusion matrix](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix). For detection runs, a per-class outcome table
leads instead, with the confusion matrix rendered below it.

#### Interactive confusion matrix — classification runs

Clicking a confusion-matrix cell shows a thumbnail gallery of example images
for that `(true_class, predicted_class)` pair, each with its confidence
score. `eval_report.json` stores up to 20 samples per cell:

```json
{
  "samples": [
    {
      "path": "cat/img_001.jpg",
      "true_class": "cat",
      "predicted_class": "dog",
      "confidence": 0.82
    }
  ]
}
```

Clicking a cell filters the gallery to samples matching that
`(true_class, predicted_class)` pair — the fastest way to see exactly which
images a model confuses, not just how many.

<!-- IMAGE PLACEHOLDER: interactive confusion matrix screenshot — see design spec for capture instructions. Suggested alt text: "CVBench WebUI confusion matrix with a clicked cell showing a gallery of misclassified sample thumbnails" -->

#### Interactive confusion matrix — detection runs

For detection, a plain N×N confusion matrix carries little signal — models
rarely confuse one class for another. The real failure modes are
localization (boxes in the wrong place) and over-prediction (spurious or
duplicate boxes), so the **headline view is a per-class outcome table**
instead, with columns:

| Outcome | Meaning |
|---|---|
| **Matched** | Ground-truth box correctly detected |
| **Mislocated** | Right class, but the predicted box misses the localization threshold |
| **Confused** | Detected, but with the wrong class |
| **Missed** | Ground-truth box with no matching prediction |

Predictions with no matching ground truth split into two chip strips:

- **Duplicate** — overlaps an already-covered ground-truth object (a
  redundant second box on the same target)
- **Spurious** — overlaps no ground-truth object of its class (a genuine
  false positive on background)

Every non-zero number in the table is clickable and filters the sample
gallery via tags like `circle:mislocated` or `square:duplicate`. The
class-confusion matrix still renders, inline, below the outcome table (not
collapsed) for cases where cross-class confusion is worth checking.

The CLI's `evaluate`/`runs show` terminal report mirrors this layout:
localization → per-class outcomes table → FP/FN reconciliation → extra
predictions → detection quality metrics (AP/precision/recall/F1) →
class-confusion matrix → class-agnostic TP/FP/FN summary.

### Compare

Pick a second run and diff the two side by side — the WebUI equivalent of the
CLI's `runs compare`.

### Export

Export the run to TFLite, ONNX, or a Hailo package, or view Jetson deployment
instructions, without leaving the browser. Mirrors `runs export` on the CLI —
see [Export Formats (TFLite/ONNX)](../deployment/export.md).

<!-- IMAGE PLACEHOLDER: WebUI run detail screenshot — see design spec for capture instructions. Suggested alt text: "CVBench WebUI run detail page showing Training, Evaluation, Compare, and Export tabs" -->

## Datasets page

Lists every dataset under `data/`, labeling each one's format
(classification or YOLO). For YOLO datasets it draws the bounding boxes over
every thumbnail:

- **Show boxes** toggles the bounding-box overlay.
- The class filter keeps only images containing a given class.

See [Generating Synthetic Data](../data/synthetic.md) for the folder layouts
this page reads.

Next: [Export Formats (TFLite/ONNX)](../deployment/export.md).
