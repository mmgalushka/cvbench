"""Evaluation API — trigger and retrieve evaluation results.

Planned endpoints
-----------------
    POST /api/evaluate/{name}         run evaluation on the test split for a given experiment
    GET  /api/evaluate/{name}         retrieve existing eval_report.json for a run
    GET  /api/runs/{name}/images/{path}  serve a test image by its relative path (needed for
                                      the interactive confusion matrix — see below)

Each handler should call:
    from cvbench.services.evaluation import run_evaluation

Interactive confusion matrix (critical UX requirement)
------------------------------------------------------
The frontend renders the confusion matrix as a clickable grid.  When the user
clicks a cell (true_class × predicted_class) it shows a thumbnail gallery of
misclassified (or correctly classified) example images for that cell.

The data is already in eval_report.json — cvbench.core.evaluator stores up to
20 samples per cell in report["samples"] with: path, true_class,
predicted_class, confidence.  See core/evaluator.py _collect_samples().

The frontend filters report["samples"] client-side by (true_class,
predicted_class) on click, then loads thumbnails via:
    GET /api/runs/{name}/images/{relative_path}

That image-serving endpoint reads from cfg.data.test_dir + relative_path and
streams the file back — no copies needed.

TODO: implement (tracked in a follow-up GitHub issue)
"""

from fastapi import APIRouter

router = APIRouter()


# @router.post("/evaluate/{name}")
# def run_eval(name: str): ...

# @router.get("/evaluate/{name}")
# def get_eval_report(name: str): ...
