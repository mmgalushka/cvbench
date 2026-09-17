# JupyterLab

For anything the CLI and WebUI don't cover — one-off experiments, poking at
a dataset interactively, or loading an already-trained model into your own
code — JupyterLab is installed in the same container.

Unlike the WebUI, it doesn't start automatically. Start it yourself, ideally
inside a [tmux session](tmux.md) so it survives closing your terminal:

```bash
tm new lab
jupyter lab --ip=0.0.0.0 --no-browser
```

Then open `http://<server-ip>:8888` in your browser.

## What you get

- The full `cvbench` package importable from a notebook — load a trained
  experiment, run predictions, or inspect a dataset with pandas/matplotlib
  directly, without going through the CLI.
- The same mounted volumes as the CLI and WebUI (`data/`, `workspace/`,
  `experiments/`) — a notebook can read a dataset or an experiment's
  checkpoint the same way the CLI does.

Use it when you want to go off-script; use the CLI or the
[Experiment Tracker](experiment-tracker.md) for the day-to-day workflow.
