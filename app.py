# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import uvicorn

from backend.run_preprocess import RunPreprocess

app = FastAPI(title="Exoplanet detection API")

# instantiate the pipeline+model once
runner = RunPreprocess()


# Pydantic models to validate input
class SeriesExample(BaseModel):
    time: List[float]
    flux: List[float]
    flux_err: Optional[List[float]] = None
    label: Optional[Any] = None


class DetectRequest(BaseModel):
    # either supply a single series:
    time: Optional[List[float]] = None
    flux: Optional[List[float]] = None
    flux_err: Optional[List[float]] = None

    # or supply multiple:
    examples: Optional[List[SeriesExample]] = None

    # or supply flat features (period/depth/duration/sn r etc.)
    # we allow arbitrary extra fields:
    class Config:
        extra = "allow"


@app.get("/")
async def index():
    return {"message": "Hello World"}


@app.post("/detect")
async def detect(payload: Dict = Field(..., description="JSON payload")):
    """
    Accepts:
    - {"time": [...], "flux": [...], "flux_err": [...]}  (single)
    - {"examples":[{"time":[...],"flux":[...]} , ... ]}  (batch)
    - or flat features: {"period":..., "depth":...}
    """
    # payload validated in RunPreprocess._make_input_df
    try:
        resp = runner.detect(payload)
        if "error" in resp:
            raise HTTPException(status_code=400, detail=resp["error"])
        return resp
    except HTTPException:
        raise
    except Exception as e:
        # unexpected
        raise HTTPException(status_code=500, detail=str(e))


# optional: run via `python app.py`
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
