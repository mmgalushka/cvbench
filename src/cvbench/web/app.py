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

Planned API endpoints (implement in separate issues)
----------------------------------------------------
    GET  /api/runs                    list all experiment runs
    GET  /api/runs/{name}             single run detail (config + metrics)
    POST /api/train                   start a training run
    GET  /api/train/{name}/status     live training status / log tail
    POST /api/evaluate/{name}         trigger evaluation on a run
    POST /api/predict                 upload image → prediction result

Running (once web extras are installed)
---------------------------------------
    pip install cvbench[web]
    serve                             # http://localhost:8000
    serve --host 0.0.0.0 --port 8080
"""

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

    if _STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    return app


def main():
    """CLI entry point: cvbench-web."""
    import argparse

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "WebUI dependencies are not installed. Run: pip install cvbench[web]"
        ) from exc

    parser = argparse.ArgumentParser(description="CVBench WebUI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    uvicorn.run("cvbench.web.app:create_app", factory=True, host=args.host, port=args.port)
