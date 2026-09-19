# Sessions (tmux)

Training and other long-running commands keep running even after you close
your terminal or lose your SSH connection — as long as you start them inside
a `tm` session. `tm` is a small wrapper around [`tmux`](https://github.com/tmux/tmux/wiki) that ships in the
container.

```
tm new <name>       new tmux session
tm connect <name>   connect / attach to session
tm delete <name>    delete session
tm list             list sessions
```

## Typical flow

```bash
tm new train
# ... now inside the tmux session ...
train data/synthetic --epochs 50
```

Detach without stopping the run with `Ctrl+B D`. The session — and whatever
is running inside it — keeps going in the background.

Later, from the same or a different terminal:

```bash
tm connect train
```

drops you back into the same session, output and all.

## Managing sessions

```bash
tm list              # see every active session
tm delete train       # stop and remove a session you no longer need
```

!!! tip
    Give sessions names that match what's running in them (`tm new train`,
    `tm new evaluate`) — `tm list` is much easier to read that way.

Next: [Experiment Tracker](experiment-tracker.md).
