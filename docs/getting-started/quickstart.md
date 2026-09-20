# Quickstart

The fastest path from a running container to a trained, evaluated, served model.

```text
commands ──▶ tm new ──▶ data generate ──▶ train ──▶ runs list ──▶ evaluate ──▶ serve
```

Five commands, top to bottom, and you have a trained model. Run these inside
the container (`docker exec -it cvbench bash`):

```text
1  commands                           # show this screen again any time
2  tm new <name>                      # start a tmux session so training survives closing your terminal
3  data generate                      # make a 4-class synthetic dataset in data/synthetic/
4  train data/synthetic --epochs 5    # train a model — prints the run name when it finishes
5  runs list                          # see every run, newest first
6  evaluate <run-name>                # score that run on the held-out test split
7  serve --host 0.0.0.0 --port 8000   # browse it all in the WebUI → http://<server-ip>:8000
```

Every command has worked examples in its `--help`. Inside the container, run
`commands` any time for the full picture (CLI plus the tmux session helpers).

For installing and starting the container itself, see
[Installation & Volumes](installation.md). For a guided, narrated walkthrough
of the same path with explanations at each step, see
[Your First Model](first-model.md).
