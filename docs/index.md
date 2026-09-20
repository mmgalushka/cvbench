---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

![CVBench](assets/images/logo-horizontal-light.svg#only-light){ .hero-logo }
![CVBench](assets/images/logo-horizontal-dark.svg#only-dark){ .hero-logo }

# <span class="sr-only">CVBench</span>

### One GPU-enabled container bundling [Keras](https://keras.io/), [TensorFlow](https://www.tensorflow.org/), [JupyterLab](https://jupyterlab.readthedocs.io/), and a WebUI for computer vision work.

No more stitching together a training script, a notebook, an experiment
tracker, and a deployment story by hand. CVBench packages all of it into one
container, so you can train, evaluate, predict, track, and deploy computer
vision models — and focus on the problem instead of managing an ML
environment.

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/mmgalushka/cvbench){ .md-button }

</div>

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } [**Train**](training/basics.md)

    ---

    Train classification and detection models with augmentation,
    optimizer/loss options, and two-phase fine-tuning built in.

-   :material-chart-box-outline:{ .lg .middle } [**Evaluate**](evaluation/evaluate.md)

    ---

    Score any run on its held-out test split — per-class metrics, confusion
    matrices, and detection outcome tables.

-   :material-target:{ .lg .middle } [**Predict**](evaluation/predict.md)

    ---

    Run inference on new images with a trained experiment, from the CLI or
    the WebUI.

-   :material-package-variant-closed:{ .lg .middle } [**Deploy**](deployment/export.md)

    ---

    Export a trained model to [TFLite](https://ai.google.dev/edge/litert), [ONNX](https://onnx.ai/), or a [Hailo](https://hailo.ai/) HEF package, or get
    step-by-step Jetson deployment instructions — all from one command.

-   :material-history:{ .lg .middle } [**Explore**](tools/experiment-tracker.md)

    ---

    Every run is recorded automatically — browse, compare, and revisit past
    experiments at any time.

-   :material-notebook-outline:{ .lg .middle } [**Customize**](tools/jupyter-notebook.md)

    ---

    Need something the CLI doesn't cover? JupyterLab is right there in the
    same container for custom, one-off experimental work.

</div>

## From zero to a model on your device

<p class="lede">No training script. No notebook. No glue code. <b>Six steps, one sandbox, zero lines of code.</b></p>

<div class="timeline" markdown="0">
<div class="step"><div class="badge">1</div><div class="step-body">
<div><h3><a href="getting-started/installation/">Deploy the sandbox</a></h3><p>One container bundles Keras, TensorFlow, JupyterLab and the WebUI.</p>
<div class="term"><span class="p">$ </span>docker run -d --gpus all \<br>&nbsp;&nbsp;-p&nbsp;8000:8000&nbsp;-p&nbsp;8888:8888&nbsp;\<br>&nbsp;&nbsp;mmgalushka/cvbench:latest</div></div>
<div class="vis"><div class="chips"><span class="chip on">Docker</span><span class="chip on">GPU</span><span class="chip">:8000 WebUI</span><span class="chip">:8888 Jupyter</span></div></div>
</div></div>
<div class="step"><div class="badge">2</div><div class="step-body">
<div><h3><a href="data/prepare/">Get a dataset</a></h3><p>Grab a cats-vs-dogs folder and split it into train / val / test.</p>
<div class="term"><span class="p">$ </span>curl -LO &lt;your-dataset-url&gt;/cats_dogs.zip<br><span class="p">$ </span>unzip cats_dogs.zip -d data/pool<br><span class="p">$ </span>data split data/pool data/cats_dogs<br><span class="p">$ </span>data list</div></div>
<div class="vis"><div class="thumbs"><figure class="tile cat"><img src="assets/images/dataset/cat1.jpg" alt="A tabby cat with blue eyes" loading="lazy"><figcaption>cat</figcaption></figure><figure class="tile cat"><img src="assets/images/dataset/cat2.jpg" alt="A ginger cat being petted" loading="lazy"><figcaption>cat</figcaption></figure><figure class="tile cat"><img src="assets/images/dataset/cat3.jpg" alt="A cat lying on its side" loading="lazy"><figcaption>cat</figcaption></figure><figure class="tile dog"><img src="assets/images/dataset/dog1.jpg" alt="A tan dog resting on grass" loading="lazy"><figcaption>dog</figcaption></figure><figure class="tile dog"><img src="assets/images/dataset/dog2.jpg" alt="A shiba inu dog yawning" loading="lazy"><figcaption>dog</figcaption></figure><figure class="tile dog"><img src="assets/images/dataset/dog3.jpg" alt="A small dog in a park" loading="lazy"><figcaption>dog</figcaption></figure></div><div class="split"><span>train 80%</span><span>val 10%</span><span>test 10%</span></div></div>
</div></div>
<div class="step"><div class="badge">3</div><div class="step-body">
<div><h3><a href="training/basics/">Train</a></h3><p>Point at the data folder; CVBench handles augmentation, loss and tracking.</p>
<div class="term"><span class="p">$ </span>train data/cats_dogs --epochs 10<br><span class="c">… epoch 10/10  acc 0.97  val_acc 0.95</span><br><span class="c"># prints the run name when it finishes</span></div></div>
<div class="vis"><svg viewBox="0 0 240 120" role="img" aria-label="Training accuracy rising and loss falling over 10 epochs">
<g stroke="var(--md-default-fg-color--lightest)" stroke-width="1"><path d="M28 10H232M28 40H232M28 70H232M28 100H232"/></g>
<path d="M28 88C70 50 110 30 150 22S210 14 232 12" fill="none" stroke="var(--cv-ok)" stroke-width="2.5"/>
<path d="M28 20C70 60 110 84 150 92S210 98 232 98" fill="none" stroke="var(--md-primary-fg-color)" stroke-width="2.5"/>
<circle cx="232" cy="12" r="4" fill="var(--cv-ok)"/><circle cx="232" cy="98" r="4" fill="var(--md-primary-fg-color)"/>
<g font-size="10" font-family="monospace" fill="var(--md-default-fg-color--light)"><text x="30" y="116">epoch 1</text><text x="200" y="116">10</text></g>
<text x="140" y="10" font-size="10" font-family="monospace" fill="var(--cv-ok)">accuracy</text><text x="140" y="84" font-size="10" font-family="monospace" fill="var(--md-primary-fg-color)">loss</text></svg></div>
</div></div>
<div class="step"><div class="badge">4</div><div class="step-body">
<div><h3><a href="evaluation/evaluate/">Evaluate</a></h3><p>Score on the held-out test split, then explore results in the WebUI.</p>
<div class="term"><span class="p">$ </span>evaluate &lt;run-name&gt;<br><span class="p">$ </span>serve --host 0.0.0.0 --port 8000</div></div>
<div class="vis"><div class="cm"><div class="h"></div><div class="h">pred cat</div><div class="h">pred dog</div><div class="h">cat</div><div class="hi">48</div><div class="lo">2</div><div class="h">dog</div><div class="lo">3</div><div class="hi">47</div></div></div>
</div></div>
<div class="step"><div class="badge">5</div><div class="step-body">
<div><h3><a href="deployment/export/">Export for your target</a></h3><p>One command; pick the format your hardware speaks.</p>
<div class="term"><span class="p">$ </span>runs export &lt;run-name&gt; --format plan</div></div>
<div class="vis"><div class="chips"><span class="chip">TFLite</span><span class="chip">ONNX</span><span class="chip">Hailo HEF</span><span class="chip on">Jetson</span></div></div>
</div></div>
<div class="step"><div class="badge">6</div><div class="step-body">
<div><h3><a href="deployment/jetson/">Run it on the device</a></h3><p>Copy the ONNX model to the Jetson and build the TensorRT engine with the printed steps.</p>
<div class="term"><span class="p">$ </span>scp .../export/plan/model.onnx user@jetson:~<br><span class="p">$ </span>trtexec --onnx=model.onnx --saveEngine=model.plan</div></div>
<div class="vis"><figure class="device"><img src="assets/images/dataset/jetson-nano.png" alt="NVIDIA Jetson Nano Developer Kit" loading="lazy"><figcaption>Jetson Nano · <span>cat 0.98</span></figcaption></figure></div>
</div></div>
</div>

<div class="zero-code"><strong>Zero lines of code.</strong> Every step above ran inside one sandbox from the CLI or WebUI. Need something custom? <a href="tools/jupyter-notebook/">JupyterLab</a> is right there in the same container.</div>

<div class="hero-footer" markdown>

Full CLI reference and guides: use the navigation above, or run `commands`
inside the container for the complete picture.

</div>
