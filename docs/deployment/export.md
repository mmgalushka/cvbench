# Export Formats (TFLite/ONNX)

`runs export` packages the best checkpoint of a run into a deployable format.

```bash
runs export my_run --format tflite                    # plain float TFLite model
runs export my_run --format tflite --quantize int8    # dynamic-range int8 TFLite (float32 I/O)
runs export my_run --format onnx                       # ONNX model for onnxruntime
runs export my_run --format hailo                       # prepare a Hailo compilation package
runs export my_run --format plan                        # print Jetson TensorRT build instructions
```

`RUN` is a run name or full path to a run directory.

| Option | Description |
|---|---|
| `--format` | `tflite` \| `onnx` \| `plan` \| `hailo` (required) |
| `--quantize` | TFLite quantization mode: `none` (default) \| `float16` \| `int8` (dynamic-range: int8 weights, float32 input/output). Ignored for other formats |
| `--output <dir>` | Parent directory for `tflite`/`onnx`; a `<format>` subfolder is created inside (default: `<experiment>/export/`). Ignored for plan and hailo |
| `--hw-arch` | Hailo target: `hailo8` \| `hailo8l` (default) \| `hailo8r`. Hailo only |
| `--calib-total N` | Target total images in the Hailo calibration set (default: `1024`) |
| `--calib-strategy` | How to distribute Hailo calibration samples — see [Hailo HEF Export](hailo.md) |

## Which format do I want?

| Target | Format | Notes |
|---|---|---|
| [onnxruntime](https://onnxruntime.ai/) / general CPU inference | `onnx` | Portable [ONNX](https://onnx.ai/), opset 13 |
| Smaller TFLite file for edge CPUs | `tflite --quantize int8` | Dynamic-range int8 weights; input/output stay float32 (not full-integer, so not for integer-only microcontrollers) |
| Mobile / edge devices, no quantization | `tflite` | Plain float [TFLite](https://ai.google.dev/edge/litert) |
| Mobile / edge devices, smaller + faster | `tflite --quantize float16` | Half the size, minimal accuracy loss |
| [Hailo](https://hailo.ai/) AI accelerator (hailo8l) | `hailo` | Produces a package for the Hailo Docker toolchain — see [Hailo HEF Export](hailo.md) |
| [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) | `plan` | Prints build steps; the `.plan` engine itself must be compiled on the Jetson — see [Jetson Deployment](jetson.md) |

!!! tip "Quantization in one paragraph"
    `float16` stores weights in half precision — about half the size with
    negligible accuracy change. `int8` shrinks the model further but is more
    sensitive to how it was trained. See TensorFlow Lite's
    [post-training quantization guide](https://ai.google.dev/edge/litert/models/post_training_quantization)
    for the trade-offs.

`onnx` and `plan` need the `export` extra: `uv pip install 'cvbench[export]'`.

## Input contract

Every exported model takes a `float32` tensor of shape `[1, size, size, 3]`
(NHWC, RGB) with pixel values in **0–255**; normalisation happens inside the
model. Minimal onnxruntime usage:

```python
import numpy as np, onnxruntime as ort

sess = ort.InferenceSession("model.onnx")
x = np.zeros((1, 224, 224, 3), dtype=np.float32)  # your image, values 0-255
probs = sess.run(None, {"image": x})[0]
```

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

## Further reading

- [TensorFlow Lite (LiteRT)](https://ai.google.dev/edge/litert) and its
  [quantization guide](https://ai.google.dev/edge/litert/models/post_training_quantization)
- [ONNX](https://onnx.ai/) and [ONNX Runtime](https://onnxruntime.ai/)
- [Technology References](../references.md) — every external tool CVBench uses

Next: [Hailo HEF Export](hailo.md) or [Jetson Deployment](jetson.md).
