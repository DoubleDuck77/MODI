from fastapi import FastAPI, HTTPException, File, UploadFile
import uvicorn
import io
import ast
import pandas as pd
from typing import Optional

from backend.run_preprocess import RunPreprocess

app = FastAPI(title="Exoplanet detection API")

# instantiate the pipeline+model once
runner = RunPreprocess()


@app.get("/")
async def index():
    return {"message": "Hello World"}


@app.post("/detect")
async def detect():
    """
    Accepts only a single multipart file upload (required) with .csv extension.
    Allowed CSV formats:
      - Long-format: series_id (or id), time, flux  (one time point per row; grouped by id)
      - One-row-per-example: columns 'time' and 'flux' where cells are JSON arrays
    """
    if file is None:
        raise HTTPException(status_code=400, detail="A CSV file must be provided in 'file' field")

    filename = file.filename or ""
    model = joblib.load("exoplanet_model.pkl")



if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
