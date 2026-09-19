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
(`keras`) prediction.

Next: package the model for deployment — see
[Export Formats (TFLite/ONNX)](../deployment/export.md),
[Hailo HEF Export](../deployment/hailo.md), or
[Jetson Deployment](../deployment/jetson.md).
