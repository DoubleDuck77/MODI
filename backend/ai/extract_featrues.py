from tsfresh import extract_features
from tsfresh.feature_extraction.settings import MinimalFCParameters
import numpy as np
import pandas as pd

def extract_features_dataframe(df, use_flux_err=False, n_jobs=4):
    """
    df: DataFrame with columns ['time', 'flux', 'flux_err', 'label']
        - each row is a series
    use_flux_err: bool
    n_jobs: int
    """

    df = df.copy()
    df['id'] = np.arange(len(df))
    
    rows = []
    for _, row in df.iterrows():
        id_ = row['id']
        t = np.asarray(row['time'])
        f = np.asarray(row['flux'])
        rows.append(pd.DataFrame({'id': id_, 'time': t, 'flux': f}))
        if use_flux_err and row['flux_err'] is not None:
            ferr = np.asarray(row['flux_err'])
            rows.append(pd.DataFrame({'id': id_, 'time': t, 'flux_err': ferr}))
    
    long_df = pd.concat(rows, ignore_index=True)
    
    feature_cols = ['flux']
    if use_flux_err:
        feature_cols.append('flux_err')
    
    feature_dfs = []
    for col in feature_cols:
        temp_df = long_df[['id', 'time', col]].rename(columns={col:'value'})
        feats = extract_features(
            temp_df,
            column_id='id',
            column_sort='time',
            column_value='value',
            disable_progressbar=False,
            n_jobs=n_jobs,
            settings=MinimalFCParameters()
        )
        feats.columns = [f"{col}__{c}" for c in feats.columns]
        feature_dfs.append(feats)
    
    features = pd.concat(feature_dfs, axis=1)
    
    labels = df.set_index('id')['label']
    final_df = features.join(labels)
    
    return final_df
