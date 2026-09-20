# Technology References

Official documentation for every external technology CVBench builds on or
exports to. Start here when you want to go deeper than these pages do.

## Framework and models

| Technology | What CVBench uses it for | Read more |
|---|---|---|
| Keras | Model definition, training, `.keras` checkpoints | [keras.io](https://keras.io/) |
| TensorFlow | Training backend | [tensorflow.org](https://www.tensorflow.org/) |
| Keras Applications | Pretrained backbones (EfficientNet, ResNet) | [Keras Applications](https://keras.io/api/applications/) |
| Transfer learning | Freeze → fine-tune workflow | [TensorFlow guide](https://www.tensorflow.org/tutorials/images/transfer_learning) |
| Optimizers and losses | `--optimizer`, `--loss` | [Adam](https://keras.io/api/optimizers/adam/), [SGD](https://keras.io/api/optimizers/sgd/), [losses](https://keras.io/api/losses/probabilistic_losses/) |
| ReduceLROnPlateau | `--lr-scheduler` | [Keras callback](https://keras.io/api/callbacks/reduce_lr_on_plateau/) |

## Export formats and runtimes

| Technology | What CVBench uses it for | Read more |
|---|---|---|
| TensorFlow Lite (LiteRT) | `--format tflite`, float16/int8 quantization | [LiteRT](https://ai.google.dev/edge/litert), [quantization](https://ai.google.dev/edge/litert/models/post_training_quantization) |
| ONNX | `--format onnx` | [onnx.ai](https://onnx.ai/) |
| ONNX Runtime | Running exported ONNX models in `predict` | [onnxruntime.ai](https://onnxruntime.ai/) |

## Edge hardware

| Technology | What CVBench uses it for | Read more |
|---|---|---|
| Hailo | `--format hailo` compilation package | [hailo.ai](https://hailo.ai/), [Developer Zone](https://hailo.ai/developer-zone/) (free account), [Model Zoo](https://github.com/hailo-ai/hailo_model_zoo), [Hailo-8L](https://hailo.ai/products/hailo-accelerators/hailo-8l-ai-accelerator/) |
| NVIDIA Jetson | `--format plan` deployment | [Jetson](https://developer.nvidia.com/embedded-computing) |
| TensorRT | Engine build on Jetson (`trtexec`) | [TensorRT](https://developer.nvidia.com/tensorrt), [docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/) |
| DeepStream | Video pipelines on Jetson | [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) |

## Data and evaluation

| Technology | What CVBench uses it for | Read more |
|---|---|---|
| YOLO label format | Detection datasets (`images/` + `labels/`) | [Ultralytics dataset format](https://docs.ultralytics.com/datasets/detect/) |
| Precision / recall / F1, confusion matrix | Evaluation reports | [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) |
| Stratified splitting | `data split` | [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) |
| k-means | Hailo calibration strategies | [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means) |

## Environment and tools

| Technology | What CVBench uses it for | Read more |
|---|---|---|
| Docker | Container runtime | [Install Docker](https://docs.docker.com/engine/install/), [Compose](https://docs.docker.com/compose/) |
| NVIDIA Container Toolkit | GPU access in the container | [Install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| tmux | Long-running sessions (`tm`) | [tmux wiki](https://github.com/tmux/tmux/wiki) |
| JupyterLab | Custom experiments | [jupyterlab.readthedocs.io](https://jupyterlab.readthedocs.io/) |
| pandas, matplotlib | Analysis in notebooks | [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/) |

## Image credits

The cat and dog photos on the home page are CC0 (public domain dedication) images from
[Wikimedia Commons](https://commons.wikimedia.org/): *Tabby cat with blue eyes*,
*Small Cat Receiving Love*, *Playing time of our cat*, *Dog resting on the grass*,
*Picography dog yawning 1*, and *Dog park, small dog mix*.

*NVIDIA Jetson Nano Developer Kit* photo by
[SparkFun Electronics](https://www.flickr.com/people/41898857@N04), via
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:NVIDIA_Jetson_Nano_Developer_Kit_(40650425503).jpg),
licensed [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).
