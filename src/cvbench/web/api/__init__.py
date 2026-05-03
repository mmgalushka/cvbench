"""CVBench REST API — assembles all sub-routers under /api.

Each sub-module owns one resource area and registers its own routes.
To add a new resource: create a new module, define a router, and include
it here.
"""

# TODO: uncomment each import as the corresponding module is implemented.
#
# from cvbench.web.api import runs, training, evaluation, prediction
#
# router.include_router(runs.router,       tags=["runs"])
# router.include_router(training.router,   tags=["training"])
# router.include_router(evaluation.router, tags=["evaluation"])
# router.include_router(prediction.router, tags=["prediction"])

try:
    from fastapi import APIRouter
    from cvbench.web.api import runs, explain, prediction, export, datasets

    router = APIRouter()
    router.include_router(runs.router,        tags=["runs"])
    router.include_router(explain.router,     tags=["explain"])
    router.include_router(prediction.router,  tags=["prediction"])
    router.include_router(export.router,      tags=["exports"])
    router.include_router(datasets.router,    tags=["datasets"])
    # router.include_router(training.router,   tags=["training"])
    # router.include_router(evaluation.router, tags=["evaluation"])
except ImportError:
    router = None  # type: ignore[assignment]  # guarded in app.create_app()
