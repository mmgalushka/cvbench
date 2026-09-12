"""``serve`` — start the CVBench WebUI."""
import os

import click

from cvbench.cli import _help


@_help.command(
    examples=[
        ("serve", "Start on http://127.0.0.1:8000 (this machine only)"),
        ("serve --host 0.0.0.0 --port 8000",
         "Accept connections from other machines — what the container does by default"),
    ],
    see_also=[("runs list", "the same runs, in the terminal")],
)
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Interface to bind. Use 0.0.0.0 to accept connections from other machines.")
@click.option("--port", default=8000, show_default=True, type=int,
              help="TCP port to listen on.")
def serve(host, port):
    """Start the CVBench WebUI server.

    Browse experiments, metrics, confusion matrices and predictions in a
    browser. Needs the [web] extras: pip install cvbench[web]

    Set the CVBENCH_URL environment variable to override the URL printed at
    startup — useful when the container is reached through a different hostname
    than the one it binds to.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise click.ClickException(
            "WebUI dependencies are not installed. Run: pip install cvbench[web]"
        ) from exc

    cvbench_url = os.environ.get("CVBENCH_URL")
    if cvbench_url:
        click.echo(f"CVBench WebUI → {cvbench_url}")
    else:
        click.echo(f"CVBench WebUI → http://{host}:{port}")
        click.echo("  Set CVBENCH_URL to override (e.g. when accessing from another machine)")

    uvicorn.run("cvbench.web.app:create_app", factory=True, host=host, port=port)
