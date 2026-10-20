"""CVBench REST API — assembles all sub-routers under /api.

Each sub-module owns one resource area and registers its own routes.
To add a new resource: create a new module, define a router, and include
it here.
"""

try:
    from fastapi import APIRouter

    from cvbench.web.api import datasets, explain, export, prediction, runs

    router = APIRouter()
    router.include_router(runs.router,        tags=["runs"])
    router.include_router(explain.router,     tags=["explain"])
    router.include_router(prediction.router,  tags=["prediction"])
    router.include_router(export.router,      tags=["exports"])
    router.include_router(datasets.router,    tags=["datasets"])
except ImportError:
    router = None  # type: ignore[assignment]  # guarded in app.create_app()
