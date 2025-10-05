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
async def detect(file: Optional[UploadFile] = File(...)):
    """
    Accepts only a single multipart file upload (required) with .csv extension.
    Allowed CSV formats:
      - Long-format: series_id (or id), time, flux  (one time point per row; grouped by id)
      - One-row-per-example: columns 'time' and 'flux' where cells are JSON arrays
    """
    if file is None:
        raise HTTPException(status_code=400, detail="A CSV file must be provided in 'file' field")

    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")

    try:
        raw = await file.read()
        text = raw.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    # Strict acceptance: either long-format with series_id/id + time + flux OR one-row-per-example arrays
    # 1) One-row-per-example with JSON arrays in time/flux columns
    if set(("time", "flux")).issubset(df.columns):
        first_time = df["time"].iloc[0]
        # detect JSON-array strings in cells
        if isinstance(first_time, str) and first_time.strip().startswith("["):
            examples = []
            for _, row in df.iterrows():
                try:
                    t = ast.literal_eval(row["time"]) if pd.notna(row["time"]) else []
                    f = ast.literal_eval(row["flux"]) if pd.notna(row["flux"]) else []
                except Exception:
                    raise HTTPException(status_code=400, detail="time/flux cells must contain JSON arrays (e.g. \"[0.0,0.02,...]\")")
                ferr = None
                if "flux_err" in df.columns and pd.notna(row.get("flux_err")):
                    try:
                        ferr = ast.literal_eval(row["flux_err"])
                    except Exception:
                        raise HTTPException(status_code=400, detail="flux_err cells must contain JSON arrays when present")
                lbl = row.get("label") if "label" in df.columns else None
                examples.append({"time": t, "flux": f, "flux_err": ferr, "label": lbl})
            input_df = pd.DataFrame(examples)

        else:
            # Could be long-format; require series_id or id for grouping
            id_col = None
            for cand in ("series_id", "id"):
                if cand in df.columns:
                    id_col = cand
                    break

            if id_col is None:
                raise HTTPException(
                    status_code=400,
                    detail=("CSV contains 'time' and 'flux' but values are not JSON arrays. "
                            "For long-format CSV you must include a 'series_id' or 'id' column to group rows.")
                )

            # Ensure time and flux convertible to float
            if not pd.api.types.is_numeric_dtype(df["time"]) and not all(isinstance(x, (int, float)) for x in df["time"].tolist()):
                # try convert
                try:
                    df["time"] = pd.to_numeric(df["time"])
                except Exception:
                    raise HTTPException(status_code=400, detail="time column must be numeric for long-format CSV")

            if not pd.api.types.is_numeric_dtype(df["flux"]):
                try:
                    df["flux"] = pd.to_numeric(df["flux"])
                except Exception:
                    raise HTTPException(status_code=400, detail="flux column must be numeric for long-format CSV")

            examples = []
            for sid, g in df.groupby(id_col):
                t = g["time"].astype(float).tolist()
                f = g["flux"].astype(float).tolist()
                ferr = None
                if "flux_err" in g.columns:
                    try:
                        ferr = g["flux_err"].astype(float).tolist()
                    except Exception:
                        raise HTTPException(status_code=400, detail="flux_err must be numeric when present in long-format CSV")
                lbl = None
                # preserve label column if present and consistent per series
                if "label" in g.columns:
                    labels = g["label"].dropna().unique().tolist()
                    lbl = labels[0] if len(labels) > 0 else None
                examples.append({"time": t, "flux": f, "flux_err": ferr, "label": lbl})
            input_df = pd.DataFrame(examples)

    # 2) Flat-feature table (flat features are NOT accepted by this strict endpoint)
    else:
        raise HTTPException(status_code=400, detail="CSV format not recognized. Only long-format (series_id/time/flux) or one-row-per-example with JSON arrays in time/flux are accepted.")

    # Now input_df is a DataFrame where each row has `time`, `flux`, `flux_err`, `label`
    try:
        resp = runner.detect(input_df)
    except Exception as e:
        # defensive: convert unexpected exceptions to 500
        raise HTTPException(status_code=500, detail=str(e))

    # runner.detect returns dicts using {"error": ...} on failure
    if isinstance(resp, dict) and "error" in resp:
        raise HTTPException(status_code=400, detail=resp["error"])

    return resp


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
