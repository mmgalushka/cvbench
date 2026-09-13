# CLI reference

The full, auto-generated CLI reference is coming in a follow-up (it will be
generated from `cvbench.cli.overview.render_markdown()` — the same source that
already keeps README.md's CLI block in sync with the Click command registry).

In the meantime:

- Run `./helper.sh docs` to regenerate the CLI reference block in
  [README.md](https://github.com/mmgalushka/cvbench#cli-reference).
- Inside the container, run `commands` for the full picture (CLI plus the tmux
  session helpers).
- Every command also documents itself: `<command> --help`.
