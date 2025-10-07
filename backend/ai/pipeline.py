import joblib
import numpy as np
import pandas as pd
from typing import Optional
from backend.ai.features import preprocess
import joblib

model = joblib.load("final_model.joblib")


def load_csv(file_path, return_dataframe: bool = True, disposition_col_name: str | None = None) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]] | pd.DataFrame:
    df = pd.read_csv(file_path)
    time = df['time'].values
    flux = df['flux'].values
    flux_err = df['flux_err'].values if 'flux_err' in df.columns else None
    disposition = df[disposition_col_name].values[0] if disposition_col_name is not None else None
    if return_dataframe:
        return pd.DataFrame({'time': time, 'flux': flux, 'flux_err': flux_err, 'disposition': disposition})
    return time, flux, flux_err, disposition


def predict_disposition(file_path: str | None, time: np.ndarray | None = None, flux: np.ndarray | None = None) -> str:
    if file_path is None and (time is None or flux is None):
        raise ValueError("Either file_path or time and flux must be provided")
    if file_path is not None:
        time, flux, _, _= load_csv(file_path, return_dataframe=False)
    features = preprocess(time=time, flux=flux)
    features.pop("disposition", None)
    features = pd.DataFrame([features]).to_numpy()[0]
    return model.predict_proba([features])[0][1]