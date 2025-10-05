# main.py (updated)
import os
import joblib
import numpy as np
import pandas as pd
from backend.ai.extract_features_custom import extract_features_dataframe


def make_toy_df(n_samples=3, series_len=50):
    """Create toy time-series dataset with columns: time, flux, flux_err, label.

    Each row represents one short time series (arrays).
    """
    rows = []
    for i in range(n_samples):
        t = np.linspace(0, 10, series_len)
        flux = np.sin(t * (i + 1)) + np.random.normal(scale=0.1, size=series_len)
        flux_err = np.full(series_len, 0.1)  # Example constant uncertainty
        label = int(i % 2)
        rows.append({
            'time': t,
            'flux': flux,
            'flux_err': flux_err,
            'label': label,
        })

    return pd.DataFrame(rows)


def align_features_to_model(X: pd.DataFrame, model, feature_list_path: str | None = None) -> pd.DataFrame:
    """Align columns of X to what the model expects.

    - If model.feature_names_in_ exists it is used.
    - Otherwise try to read feature names from the booster (XGBoost) or from
      an external feature list file (joblib dump) if provided.
    - Missing features are filled with 0; extra columns are dropped.
    """
    expected = None

    # 1) sklearn-style attribute
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)

    # 2) XGBoost booster feature names
    if expected is None:
        try:
            booster = model.get_booster()
            if booster is not None and getattr(booster, "feature_names", None):
                expected = list(booster.feature_names)
        except Exception:
            pass

    # 3) pipeline last step
    if expected is None and hasattr(model, "steps"):
        last = model.steps[-1][1]
        if hasattr(last, "feature_names_in_"):
            expected = list(last.feature_names_in_)

    # 4) external feature list (if provided)
    if expected is None and feature_list_path and os.path.exists(feature_list_path):
        try:
            expected = joblib.load(feature_list_path)
            expected = list(expected)
        except Exception:
            expected = None

    if expected is None:
        raise RuntimeError(
            "Cannot determine expected feature names from the model. "
            "Provide a saved feature-list (joblib) or train the model with a pipeline that preserves feature_names_in_."
        )

    # Drop obvious raw columns that shouldn't be passed to model
    X = X.drop(columns=[c for c in ['time', 'flux', 'flux_err'] if c in X.columns], errors='ignore')

    # Coerce to numeric
    X = X.apply(pd.to_numeric, errors='coerce')

    # Fill missing with 0
    X_aligned = X.reindex(columns=expected, fill_value=0)

    # Final dtype sanity: cast to float
    X_aligned = X_aligned.astype(float)

    return X_aligned


def get_prediction_probabilities(model, X_aligned: pd.DataFrame):
    """Return a (n_samples, n_classes) probability array and class labels list.

    Tries several strategies in order:
    1. model.predict_proba(X)
    2. model.decision_function(X) -> convert via sigmoid (binary) or softmax (multiclass)
    3. XGBoost booster predict with output_margin -> sigmoid/softmax

    Raises when none are available.
    """
    # 1) predict_proba
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_aligned)
        # for sklearn estimators, class labels available
        classes = getattr(model, 'classes_', None)
        return probs, classes

    # 2) decision_function
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X_aligned)
        # binary: scores shape (n_samples,) or (n_samples,1)
        if scores.ndim == 1 or (scores.ndim == 2 and scores.shape[1] == 1):
            # sigmoid
            s = scores.ravel()
            probs_pos = 1.0 / (1.0 + np.exp(-s))
            probs = np.vstack([1 - probs_pos, probs_pos]).T
            classes = [0, 1]
            return probs, classes
        else:
            # multiclass scores -> softmax
            exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exps / exps.sum(axis=1, keepdims=True)
            classes = list(range(probs.shape[1]))
            return probs, classes

    # 3) Try XGBoost booster raw output
    try:
        booster = model.get_booster()
        import xgboost as xgb
        dmat = xgb.DMatrix(X_aligned)
        # raw margin predictions
        margin = booster.predict(dmat, output_margin=True)
        # margin shape: (n_samples,) for binary, (n_samples, n_classes) for multi
        arr = np.array(margin)
        if arr.ndim == 1:
            probs_pos = 1.0 / (1.0 + np.exp(-arr))
            probs = np.vstack([1 - probs_pos, probs_pos]).T
            classes = [0, 1]
            return probs, classes
        else:
            exps = np.exp(arr - np.max(arr, axis=1, keepdims=True))
            probs = exps / exps.sum(axis=1, keepdims=True)
            classes = list(range(probs.shape[1]))
            return probs, classes
    except Exception:
        pass

    raise RuntimeError('Model does not support probability prediction or decision scores.')


if False == "__main__":
    # 1) Build toy data
    df = make_toy_df(n_samples=5, series_len=80)
    # flux = np.sin(t * (i + 1)) + np.random.normal(scale=0.5, size=series_len)
    # flux_err = np.random.uniform(0.05, 0.15, size=series_len)
    # 2) Extract features (this function should be the corrected version that returns
    #    a DataFrame indexed by id and optionally contains a 'label' column)
    feats_df = extract_features_dataframe(df, use_flux_err=True, n_jobs=4)
    print("Extracted features shape:", feats_df.shape)

    # 3) Prepare X and y
    X = feats_df.drop(columns=['label'], errors='ignore').copy()
    y = feats_df['label'] if 'label' in feats_df.columns else None

    # 4) Load model
    model_path = "stack_model.joblib"
    feature_list_path = "feature_names.joblib"  # optional: saved during training

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    print("Loaded model:", type(model))

    # 5) Align features to what model expects
    try:
        X_aligned = align_features_to_model(X, model)
    except RuntimeError as e:
        # If we cannot find expected features, print diagnostics and exit gracefully
        print("ERROR: could not determine model's expected features:", str(e))
        print("Available columns in X:", list(X.columns)[:30])
        raise

    print("Aligned feature matrix shape:", X_aligned.shape)

    # 6) Predict probabilities and labels with safety and helpful error messages
    try:
        probs, classes = get_prediction_probabilities(model, X_aligned)
    except Exception as exc:
        print("Probability prediction failed with:", repr(exc))
        raise

    # Determine positive-class probability column. If binary, take class 1 if present.
    if classes is None:
        # fallback: assume second column is positive
        pos_idx = 1 if probs.shape[1] > 1 else 0
    else:
        # if classes contains 1 use that index
        if 1 in list(classes):
            pos_idx = list(classes).index(1)
        else:
            # take the last column as "positive" by convention
            pos_idx = probs.shape[1] - 1

    # Add probability columns to feats_df
    # create human-friendly column names using class labels
    for i, cls in enumerate(classes if classes is not None else range(probs.shape[1])):
        col_name = f"prob_class_{cls}"
        feats_df[col_name] = probs[:, i]

    # Add aggregated single probability (positive class)
    feats_df['prob_positive'] = probs[:, pos_idx]

    # Predicted label (in case model.predict exists and is reliable)
    try:
        preds = model.predict(X_aligned)
    except Exception:
        # fallback to argmax of probs
        preds = np.argmax(probs, axis=1)

    feats_df['prediction'] = preds

    # Global probability (mean of positive-class probabilities)
    global_prob = float(np.nanmean(feats_df['prob_positive'].values))

    print("Preview of predictions & probabilities:")
    display_cols = ['prediction', 'prob_positive'] + [c for c in feats_df.columns if c.startswith('prob_class_')][:5]
    # print(feats_df[display_cols].head())
    print(f"Global positive-class probability (mean): {global_prob:.6f}")

    # # 8) Save results
    # out_path = "extracted_features_with_preds.csv"
    # feats_df.to_csv(out_path, index=True)
    # print("Saved results to", out_path)
# --- Load a single light curve CSV ---
csv_path = "KOI_lightcurves_fin/757450.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

# Expecting CSV with columns like: time, flux, flux_err (or similar)
df_input = pd.read_csv(csv_path)
print("Loaded light curve CSV:", df_input.shape)

# Wrap single curve into expected DataFrame format for extract_features_dataframe
# It expects columns: 'time', 'flux', 'flux_err', optionally 'label'
df_input_wrapped = pd.DataFrame([{
    'time': df_input['time'].values,
    'flux': df_input['flux'].values,
    'flux_err': df_input['flux_err'].values if 'flux_err' in df_input.columns else np.full(len(df_input), 0.1),
    'label': 0,  # dummy label
}])

# --- Extract features ---
feats_input = extract_features_dataframe(df_input_wrapped, use_flux_err=True, n_jobs=4)
print("Extracted features for CSV:", feats_input.shape)

# --- Align features ---
X_input = feats_input.drop(columns=['label'], errors='ignore')
model_path = "stack_model.joblib"


if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = joblib.load(model_path)
X_input_aligned = align_features_to_model(X_input, model, feature_list_path=feature_list_path)

# --- Predict probabilities ---
probs_input, classes_input = get_prediction_probabilities(model, X_input_aligned)

# Determine positive-class probability index
if 1 in list(classes_input):
    pos_idx = list(classes_input).index(1)
else:
    pos_idx = probs_input.shape[1] - 1

# Print results
print("Predicted probabilities:", probs_input)
print("Positive-class probability:", probs_input[:, pos_idx])