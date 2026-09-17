# Browsing Runs & Datasets

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

Overall accuracy and a per-class precision/recall/F1 table. For
classification runs, an interactive confusion matrix — see
[Interactive Confusion Matrix](confusion-matrix.md). For detection runs, a
per-class outcome table leads instead, with the confusion matrix rendered
below it.

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

Next: [Interactive Confusion Matrix](confusion-matrix.md).
