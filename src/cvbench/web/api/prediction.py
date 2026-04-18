"""Prediction API — single-image inference via file upload.

Planned endpoints
-----------------
    POST /api/predict    upload an image + specify a checkpoint → get predicted class + confidence

Each handler should call:
    from cvbench.services.prediction import run_prediction

The request body should be multipart/form-data:
    - file:       the image to classify (UploadFile)
    - checkpoint: path to the .keras model file

TODO: implement (tracked in a follow-up GitHub issue)
"""

from fastapi import APIRouter

router = APIRouter()


# @router.post("/predict")
# def predict(file: UploadFile, checkpoint: str): ...
