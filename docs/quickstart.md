# Quickstart

The fastest path from a running container to a trained, evaluated, served model.

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

For installing and starting the container itself, see the
[README's Quick start section](https://github.com/mmgalushka/cvbench#quick-start).

!!! note
    This page is copied from the CLI-generated Quickstart block. A follow-up will
    wire it (and the CLI reference) to generate live from `cvbench.cli.overview`
    instead of being hand-copied.
