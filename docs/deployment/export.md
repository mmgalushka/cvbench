# Export Formats (TFLite/ONNX)

`runs export` packages the best checkpoint of a run into a deployable format.

```bash
runs export my_run --format tflite                    # plain float TFLite model
runs export my_run --format tflite --quantize int8    # quantized TFLite for microcontrollers
runs export my_run --format onnx                       # ONNX model for onnxruntime
runs export my_run --format hailo                       # prepare a Hailo compilation package
runs export my_run --format plan                        # print Jetson TensorRT build instructions
```

`EXPERIMENT` is a run name or full path to a run directory.

| Option | Description |
|---|---|
| `--format` | `tflite` \| `onnx` \| `plan` \| `hailo` (required) |
| `--quantize` | TFLite quantization mode: `none` (default) \| `float16` \| `int8`. Ignored for ONNX and plan |
| `--output <dir>` | Output directory (default: `<experiment>/export/`). Ignored for plan |
| `--calib-total N` | Target total images in the Hailo calibration set (default: `1024`) |
| `--calib-strategy` | How to distribute Hailo calibration samples — see [Hailo HEF Export](hailo.md) |

## Which format do I want?

| Target | Format | Notes |
|---|---|---|
| onnxruntime / general CPU inference | `onnx` | Portable, opset 13 |
| Microcontrollers / edge CPUs | `tflite --quantize int8` | Smallest, needs a calibration-free int8 path |
| Mobile / edge devices, no quantization | `tflite` | Plain float TFLite |
| Mobile / edge devices, smaller + faster | `tflite --quantize float16` | Half the size, minimal accuracy loss |
| Hailo AI accelerator (hailo8l) | `hailo` | Produces a package for the Hailo Docker toolchain — see [Hailo HEF Export](hailo.md) |
| NVIDIA Jetson | `plan` | Prints build steps; the `.plan` engine itself must be compiled on the Jetson — see [Jetson Deployment](jetson.md) |

Each export writes `export_info.json` alongside the model file, recording the
source checkpoint, input shape/dtype, class list, backbone, and validation
metrics at export time.

```bash
predict my_run images/ --format tflite
```

verifies the exported model runs and predicts as expected — see
[Running Predictions](../evaluation/predict.md), and
`predict my_run images/ --format all` to compare every format side by side
and catch conversion drift.

Next: [Hailo HEF Export](hailo.md) or [Jetson Deployment](jetson.md).
