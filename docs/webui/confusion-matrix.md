# Interactive Confusion Matrix

## Classification runs

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

## Detection runs

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

Next: [Export Formats (TFLite/ONNX)](../deployment/export.md).
