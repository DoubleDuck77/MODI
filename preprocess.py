# backend/run_preprocess.py
import os
from functools import lru_cache
import traceback
import joblib
import numpy as np
import pandas as pd

# import your own pipeline functions here
# from extract_featrues import extract_features_dataframe
# from preprocess import pipeline_df

# If your module names differ, adjust the imports accordingly:
# from extract_featrues import extract_features_dataframe
# from preprocess import pipeline_df

# For the example I'll assume you have pipeline_df and extract_features_dataframe
# available from your preprocess / extract_featrues modules:
try:
    from preprocess import pipeline_df
    from extract_featrues import extract_features_dataframe
except Exception:
    # If your file/module names are different, make sure to adapt these imports.
    pipeline_df = None
    extract_features_dataframe = None

MODEL_PATH = os.environ.get("MODEL_PATH", "stack_model.joblib")


@lru_cache(maxsize=1)
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)


def infer_model_feature_names(model, fallback=None):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        final = list(model.named_steps.values())[-1]
        if hasattr(final, "feature_names_in_"):
            return list(final.feature_names_in_)
    for attr in ("columns_", "feature_names_", "feature_names"):
        if hasattr(model, attr):
            try:
                return list(getattr(model, attr))
            except Exception:
                pass
    return fallback


def align_features(X_df: pd.DataFrame, model):
    model_cols = infer_model_feature_names(model, fallback=X_df.columns.tolist())
    if model_cols is None:
        return X_df.values, X_df.columns.tolist()

    missing = [c for c in model_cols if c not in X_df.columns]
    for c in missing:
        X_df[c] = 0.0

    X_aligned = X_df[model_cols].copy().fillna(0.0)
    return X_aligned.values, model_cols


def predict_probs(X_df: pd.DataFrame, model):
    X_vals, cols = align_features(X_df, model)
    # Try predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_vals)
        if probs.ndim == 2:
            # choose class index for positive class
            cls_idx = 1 if getattr(model, "classes_", None) is None or 1 in list(getattr(model, "classes_", [])) else min(1, probs.shape[1]-1)
            # If classes_ present, try to pick index of positive class 1
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    cls_idx = classes.index(1)
                else:
                    # fallback to last column
                    cls_idx = probs.shape[1] - 1
            return probs[:, cls_idx].astype(float).tolist()
        return probs.ravel().astype(float).tolist()
    # fallback: decision_function -> sigmoid
    if hasattr(model, "decision_function"):
        scores = np.array(model.decision_function(X_vals)).ravel()
        probs = 1.0 / (1.0 + np.exp(-scores))
        return probs.astype(float).tolist()
    # final fallback: predict -> 0/1 map to floats
    preds = np.array(model.predict(X_vals)).ravel()
    return preds.astype(float).tolist()


class RunPreprocess:
    """
    Thin wrapper to run your pipeline and then model prediction.
    Methods:
      - detect(payload): payload may be a dict with either:
          * {"time": [...], "flux":[...], "flux_err":[...]}  (single series)
          * {"examples":[ {...}, {...} ] }  (multiple series)
          * or a flat dict of already-extracted features (period/depth/...)
    """
    def __init__(self, model_path=MODEL_PATH):
        self.model = load_model(model_path)

    def _make_input_df(self, payload: dict) -> pd.DataFrame:
        """
        Convert payload to a DataFrame where each row is a time-series (columns: time, flux, flux_err, label)
        or return a flat features DataFrame if payload looks like already-extracted features.
        """
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a JSON object")

        # detect flat feature dict (heuristic)
        if any(k in payload for k in ("period", "depth", "duration", "snr")):
            # single-row features DataFrame
            return pd.DataFrame([payload])

        examples = []
        if "examples" in payload and isinstance(payload["examples"], list):
            examples = payload["examples"]
        elif "time" in payload and "flux" in payload:
            examples = [payload]
        else:
            raise ValueError("Unsupported input. Provide {time:[...],flux:[...]} or {examples:[...]} or flat features")

        rows = []
        for ex in examples:
            t = np.asarray(ex.get("time", []), dtype=float)
            f = np.asarray(ex.get("flux", []), dtype=float)
            ferr = np.asarray(ex.get("flux_err", []), dtype=float) if ex.get("flux_err") is not None else None
            lbl = ex.get("label", None)
            rows.append({"time": t, "flux": f, "flux_err": ferr, "label": lbl})
        return pd.DataFrame(rows)

    def detect(self, payload: dict):
        """
        Run preprocess -> features -> predict and return probabilities and optional meta.
        """
        try:
            input_df = self._make_input_df(payload)

            # If it has 'time' and 'flux' -> run pipeline_df to preprocess & extract features
            if set(("time", "flux")).issubset(set(input_df.columns)):
                if pipeline_df is None:
                    raise RuntimeError("pipeline_df is not available — import your preprocessing pipeline correctly.")
                feat_df = pipeline_df(input_df, dt=0.0204, snr_thresh=12, inject_transits=False)
                # The returned feat_df likely has label column; drop it before predict
                feat_df = feat_df.reset_index(drop=True).drop(columns=["label"], errors="ignore")
            else:
                # Already a flat features DataFrame
                feat_df = input_df

            if feat_df.shape[0] == 0:
                return {"error": "No valid examples after preprocessing", "predictions": []}

            probs = predict_probs(feat_df, self.model)
            # shape into list of dicts
            out = [{"index": i, "probability": float(p)} for i, p in enumerate(probs)]
            if len(out) == 1:
                return {"probability": out[0]["probability"], "raw": out}
            return {"predictions": out}
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}
