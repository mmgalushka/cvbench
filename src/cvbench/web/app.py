"""CVBench WebUI — application factory.

Architecture
------------
The web layer is split into two fully decoupled concerns:

                        ┌──────────────────────────────────────────────┐
                        │                 cvbench.web                   │
                        │                                               │
                        │  ┌─────────────────┐   ┌──────────────────┐  │
                        │  │   web/api/       │   │  web/static/     │  │
                        │  │   JSON REST API  │   │  HTML + CSS + JS │  │
                        │  │   prefix: /api   │   │  mounted at: /   │  │
                        │  └────────┬─────────┘   └────────┬─────────┘  │
                        │           │  calls                │ fetch()    │
                        └───────────┼───────────────────────┼────────────┘
                                    │                        │
                        ┌───────────▼────────────────────────┘
                        │        cvbench.services
                        │  training · evaluation · prediction
                        └────────────────────────────────────

Rules that must stay true as the WebUI grows
--------------------------------------------
1. /api/* routes ALWAYS return JSON — never HTML.
   The frontend (static/) is the only thing that renders HTML.

2. /api/* routes call cvbench.services.* — never cvbench.core.* directly.
   Services are the single orchestration layer shared with the CLI.

3. The static frontend talks to the backend exclusively via fetch("/api/...").
   No server-side template rendering; no mixed routes.

4. Adding a new feature means:
     a. Add/extend a service in cvbench.services.*
     b. Expose it via a new /api/* route
     c. Call it from the frontend JS
   The CLI gets the same feature for free by calling the same service.

API routes (one module per resource under web/api/)
-----------------------------------------------------
    runs.py        GET /api/present, /api/runs, /api/runs/{name}, /api/sweeps,
                   /api/sweeps/{name}, /api/runs/{name}/images/{path};
                   PATCH, DELETE /api/runs/{name}
    prediction.py  GET /api/augmentations; POST /api/predict/{single,augmented,xai}
    explain.py     POST /api/runs/{name}/explain/gradcam
    export.py      GET, POST /api/runs/{name}/exports; download and DELETE of one export
    datasets.py    GET /api/datasets and its images; POST, DELETE dataset images

Training, evaluation and sweeps are not exposed yet. They report progress as
cvbench.core.events events through a ``listener`` argument, which a job runner
here can queue for live updates.

Running (once web extras are installed)
---------------------------------------
    pip install cvbench[web]
    serve                             # http://localhost:8000
    serve --host 0.0.0.0 --port 8080  # see: serve --help
"""

import logging
from pathlib import Path

_STATIC_DIR = Path(__file__).parent / "static"


def create_app():
    """Application factory — returns a FastAPI app instance.

    Requires the [web] optional dependencies:
        pip install cvbench[web]
    """
    try:
        import fastapi
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise ImportError(
            "WebUI dependencies are not installed. Run: pip install cvbench[web]"
        ) from exc

    from cvbench.web.api import router as api_router

    app = fastapi.FastAPI(title="CVBench", docs_url="/api/docs", redoc_url=None)
    app.include_router(api_router, prefix="/api")

    from fastapi import Request
    from fastapi.responses import JSONResponse

    from cvbench.core.errors import AmbiguousNameError, ConflictError, CVBenchError, NotFoundError

    @app.exception_handler(CVBenchError)
    async def _domain_error(_request: Request, exc: CVBenchError):
        """The one place domain errors become HTTP responses (message + candidate names)."""
        if isinstance(exc, NotFoundError):
            status = 404
        elif isinstance(exc, (AmbiguousNameError, ConflictError)):
            status = 409
        else:
            status = 400
        body: dict = {"detail": exc.message}
        candidates = getattr(exc, "candidates", None)
        if candidates:
            body["candidates"] = candidates
        return JSONResponse(body, status_code=status)

    @app.exception_handler(Exception)
    async def _unexpected_error(_request: Request, exc: Exception):
        """Anything that is not a domain error is a bug: log it and answer 500."""
        logging.getLogger("cvbench.web").exception("Unhandled error in %s", _request.url.path)
        return JSONResponse({"detail": str(exc)}, status_code=500)

    if _STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    return app


def main():
    """Backwards-compatible shim — the ``serve`` command now lives in cvbench.cli.serve."""
    from cvbench.cli.serve import serve

    serve()
