# backend/ai/extract_features_custom.py
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import MinimalFCParameters
import numpy as np
import pandas as pd
from typing import Optional

def extract_features_dataframe(df: pd.DataFrame, use_flux_err: bool = False, n_jobs: int = 4) -> pd.DataFrame:
    """
    df: DataFrame with columns ['time', 'flux', 'flux_err', 'label'] where each row is a series.
    If 'label' is missing, the returned DataFrame will have no 'label' column.
    """
    df = df.copy().reset_index(drop=True)
    df['id'] = np.arange(len(df))

    rows = []
    for _, row in df.iterrows():
        id_ = int(row['id'])
        t = np.asarray(row['time'])
        f = np.asarray(row['flux'])

        rows.append(pd.DataFrame({'id': id_, 'time': t, 'flux': f}))

        if use_flux_err:
            ferr = row.get('flux_err', None)
            if ferr is not None:
                ferr = np.asarray(ferr)
                rows.append(pd.DataFrame({'id': id_, 'time': t, 'flux_err': ferr}))

    # concat after we've collected all series
    if len(rows) == 0:
        # Nothing to extract
        empty = pd.DataFrame(index=df['id'])
        if 'label' in df.columns:
            empty = empty.join(df.set_index('id')['label'])
        return empty

    long_df = pd.concat(rows, ignore_index=True)

    feature_cols = ['flux']
    if use_flux_err:
        feature_cols.append('flux_err')

    feature_dfs = []
    for col in feature_cols:
        temp_df = long_df[['id', 'time', col]].rename(columns={col: 'value'})

        # Drop NaNs
        temp_df = temp_df.dropna(subset=['value'])

        # Remove IDs that now have no samples
        temp_df = temp_df.groupby('id').filter(lambda g: g['value'].notna().any())

        if temp_df.empty:
            # skip this feature column entirely
            continue

        # ensure numeric values (coerce if they are strings)
        temp_df['value'] = pd.to_numeric(temp_df['value'], errors='coerce')
        temp_df = temp_df.dropna(subset=['value'])
        if temp_df.empty:
            continue

        feats = extract_features(
            temp_df,
            column_id='id',
            column_sort='time',
            column_value='value',
            disable_progressbar=False,
            n_jobs=n_jobs,
            default_fc_parameters=MinimalFCParameters()
        )

        # ensure index is integer id
        feats.index = feats.index.astype(int)

        feats.columns = [f"{col}__{c}" for c in feats.columns]
        feature_dfs.append(feats)

    # Combine feature sets. If none produced, create empty frame indexed by ids
    if feature_dfs:
        features = pd.concat(feature_dfs, axis=1)
    else:
        features = pd.DataFrame(index=df['id'].astype(int))

    # join labels if present
    if 'label' in df.columns:
        labels = df.set_index('id')['label']
        final_df = features.join(labels)
    else:
        final_df = features

    # Sort index by id for predict alignment
    final_df = final_df.sort_index()

    return final_df
