# Running Predictions

`predict` runs inference on `INPUT` using a trained `EXPERIMENT`.

```bash
predict cls_effnet_b0_2026_01_21 photo.jpg
```

`EXPERIMENT` is a run name or full path to a run directory. `INPUT` is an
image file or a folder of images. Both arguments are optional when
`--format plan` is used.

## Formats

```bash
predict cls_effnet_b0_2026_01_21 images/ --format tflite
predict cls_effnet_b0_2026_01_21 images/ --format all
predict --format plan
```

| `--format` | Description |
|---|---|
| `keras` (default) | Run inference with the original [Keras](https://keras.io/) model |
| `onnx` | Run inference with the exported [ONNX](https://onnx.ai/) model (via [ONNX Runtime](https://onnxruntime.ai/)) |
| `tflite` | Run inference with the exported [TFLite](https://ai.google.dev/edge/litert) model |
| `all` | Run every available format side by side and flag where predictions disagree — useful for spotting conversion drift after export |
| `plan` | Print the Jetson inference script and run instructions (no run or images needed) |

The model for `onnx`/`tflite` must already be exported — see
[Export Formats (TFLite/ONNX)](../deployment/export.md). `predict --format
plan` doesn't run inference itself; it prints a ready-to-use [TensorRT](https://developer.nvidia.com/tensorrt)
inference script for the Jetson, since `.plan` engines only run on the device
they were compiled for.

```bash
predict cls_effnet_b0_2026_01_21 images/ --format all
```

produces a side-by-side table with each format's prediction and confidence,
flagging (⚠️) any row where a format's class disagrees with the reference
(`keras`) prediction. The `max|diff|` column is the largest probability
difference between any format and the reference, which catches drift that
leaves the top-1 class unchanged. If the reference model's mean top-1
confidence is near chance, a warning notes that matching class names are weak
evidence and `max|diff|` is the number to trust.

## More than the top class

```bash
predict cls_effnet_b0_2026_01_21 photo.jpg --top-k 3
predict cls_effnet_b0_2026_01_21 photo.jpg --probs
```

`--top-k N` shows the N most likely classes per image; `--probs` shows every
class probability. Both are classification only.

## Choosing the model file

After the results, `predict` prints which model file each format used. To run
a specific file (a quantized or relocated export, for example), pass it
directly; the format is taken from its extension (`.keras`, `.onnx`,
`.tflite`):

```bash
predict cls_effnet_b0_2026_01_21 images/ --model out/model_int8.tflite
```

`--model` can't be combined with `--format all`. For `--format tflite`, any
`.tflite` file under the run's `export/` folder is found automatically.

## Unreadable inputs

Files that can't be decoded as images are skipped and listed under the
results (`unreadable: fake.jpg  Cannot read image (...)`). If nothing in
`INPUT` is readable, `predict` exits with `Cannot read image: ...`. When no
supported images are found, the error lists the accepted extensions.

## Detection runs

For a detection run, `predict` decodes the model output and prints every box
per image: class, score, and normalized `x y w h` (top-left corner and size,
as fractions of the image).

```bash
predict det_yolo_2026_01_21 photo.jpg --conf 0.4
```

`--conf` sets the minimum score for a box to be reported; it defaults to the
run's `detection.conf_threshold` and is rejected for classification runs. With
`--format all`, each format shows its box count and top score per image, and
flags (⚠️) rows where a format's boxes differ from the `keras` reference.

Next: package the model for deployment — see
[Export Formats (TFLite/ONNX)](../deployment/export.md),
[Hailo HEF Export](../deployment/hailo.md), or
[Jetson Deployment](../deployment/jetson.md).
