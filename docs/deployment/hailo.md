# Hailo HEF Export

[Hailo](https://hailo.ai/) makes low-power AI accelerator chips (e.g. the
Hailo-8/[Hailo-8L](https://hailo.ai/products/hailo-accelerators/hailo-8l-ai-accelerator/)) used in edge and embedded devices; they run models
compiled to Hailo's own `.hef` (Hailo Executable Format) binary, not
[TFLite](https://ai.google.dev/edge/litert) or [ONNX](https://onnx.ai/)
directly. Compiling to HEF requires Hailo's own SDK/Docker toolchain (the
`hailo` CLI used below), which CVBench does not bundle — get it from the
[Hailo Developer Zone](https://hailo.ai/developer-zone/) before the commands
on this page will run.

!!! info "Hailo Developer Zone account"
    The Dataflow Compiler download and its user guide are behind a free
    Developer Zone login. Register once, then follow Hailo's installation
    guide for the Docker-based toolchain.

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
| `stratified` (default) | Equal quota per class, then [k-means](https://scikit-learn.org/stable/modules/clustering.html#k-means) clustering *within* each class to pick a diverse representative from every cluster — recommended |
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

!!! tip "What calibration is for"
    Hailo converts float32 weights and activations to low-bit integers
    (post-training quantization). It runs the calibration images through the
    model to measure the value ranges it needs to preserve, so a calibration
    set that resembles your real inputs directly affects accuracy after
    conversion.

The final calibration array is shuffled before being written — sequential
per-class ordering can bias Hailo's calibration algorithm, which processes
images in mini-batches.

## `model.alls`

The generated Model Script (`.alls`, Hailo's optimization script format —
see the Dataflow Compiler user guide for every command) sets a default optimization level and per-layer
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

## Further reading

- [Hailo Developer Zone](https://hailo.ai/developer-zone/) — Dataflow Compiler
  download, user guide, and Model Script reference (free account required)
- [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) — reference
  models and compile recipes from Hailo
- [Hailo-8L accelerator](https://hailo.ai/products/hailo-accelerators/hailo-8l-ai-accelerator/) — hardware overview
- [TensorFlow Lite (LiteRT)](https://ai.google.dev/edge/litert) — the model
  format Hailo's parser reads
- [Technology References](../references.md) — every external tool CVBench uses

Next: [Jetson Deployment](jetson.md).
