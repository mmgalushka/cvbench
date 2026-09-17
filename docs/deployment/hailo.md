# Hailo HEF Export

[Hailo](https://hailo.ai/) makes low-power AI accelerator chips (e.g. the
Hailo-8/Hailo-8L) used in edge and embedded devices; they run models compiled
to Hailo's own `.hef` (Hailo Executable Format) binary, not TFLite or ONNX
directly. Compiling to HEF requires Hailo's own SDK/Docker toolchain (the
`hailo` CLI used below), which CVBench does not bundle — check Hailo's
developer site/documentation for how to obtain and install it before the
commands on this page will run.

```bash
runs export my_run --format hailo
```

Prepares a Hailo compilation package under `experiments/my_run/export/hailo/`:

- `model.tflite` — float32 TFLite conversion of the best checkpoint
- `calib_set.npy` — a calibration image array built from the training (or
  validation) split
- `model.alls` — a Hailo Model Script with default optimization settings
- `export_info.json` — source checkpoint, input shape, classes, backbone,
  validation metrics, and calibration metadata

Existing `model.tflite`/`calib_set.npy` files are reused on a re-run rather
than rebuilt.

## Calibration strategies

`--calib-strategy` controls how calibration images are chosen from the
dataset:

| Strategy | Description |
|---|---|
| `stratified` (default) | Equal quota per class, then k-means clustering *within* each class to pick a diverse representative from every cluster — recommended |
| `proportional` | Samples proportional to each class's size in the dataset |
| `equal` | Same fixed number of images per class |
| `diverse` | k-means clustering across *all* images regardless of class, then samples spread across clusters |

```bash
runs export my_run --format hailo --calib-total 512 --calib-strategy diverse
```

| Option | Default | Description |
|---|---|---|
| `--calib-total N` | `1024` | Target total images in the calibration set |
| `--calib-strategy` | `stratified` | See table above |

The final calibration array is shuffled before being written — sequential
per-class ordering can bias Hailo's calibration algorithm, which processes
images in mini-batches.

## `model.alls`

The generated Model Script sets a default optimization level and per-layer
precision hints:

```text
model_optimization_flavor(optimization_level=2, compression_level=1)
quantization_param(avgpool1, precision_mode=a16_w16)
quantization_param(avgpool3, precision_mode=a16_w16)
quantization_param(fc11, precision_mode=a16_w16)
```

Edit this file before compiling if a specific layer needs different
precision handling for your model.

## Converting to HEF

The Hailo conversion commands must be run inside the Hailo Docker container.
Mount or copy the export folder into it: `experiments/my_run/export/hailo/`.

**Step 1 — parse TFLite to HAR:**

```bash
hailo parser tf model.tflite
```

**Step 2 — optimize with calibration data:**

```bash
hailo optimize \
    --hw-arch hailo8l \
    --calib-set-path calib_set.npy \
    --model-script model.alls \
    --output-har-path model_optimized.har \
    model.har
```

**Step 3 — compile to HEF:**

```bash
hailo compiler --hw-arch hailo8l model_optimized.har
```

!!! warning
    Hailo quantization is sensitive to how the source model was fine-tuned.
    Unfreezing the entire backbone at the same learning rate as a randomly
    initialized head can distort pretrained weight distributions enough that
    Hailo's post-training quantization fails to converge. If you hit
    conversion or SNR problems, try [Two-Phase Training](../training/two-phase.md)
    with a much lower fine-tuning learning rate before re-exporting.

Next: [Jetson Deployment](jetson.md).
