# predict.py
# ---------------------------------------
# Backend-ready inference for your 3-style models.
# Works as:
#   1) import:  svc = StyleInferenceService(model_dir="/content/outputs"); svc.predict(df)
#   2) API:     uvicorn predict:app --host 0.0.0.0 --port 8000
# ---------------------------------------

from __future__ import annotations

import json
import os
import typing as T
import warnings

import joblib
import numpy as np
import pandas as pd

# Optional: FastAPI for serving
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False


TAGS = ("visual_verbal", "active_reflective", "global_sequential")

DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", "/content/outputs")
DEFAULT_SCHEMA    = os.environ.get("SCHEMA_PATH", os.path.join(DEFAULT_MODEL_DIR, "feature_columns.json"))


class StyleInferenceService:
    """
    Loads preprocessor + per-tag calibrators and provides batch prediction.

    Expects these files in model_dir:
      - preprocessor.joblib              (sklearn ColumnTransformer)
      - feature_columns.json             (ordered list of feature names)
      - calibrator_{tag}.joblib          (CalibratedClassifierCV for each tag)
        * If a calibrator is missing, falls back to model_{tag}.json (XGBoost Booster)
          and returns its raw logistic output (already in [0,1]).

    Usage:
      svc = StyleInferenceService(model_dir="/content/outputs")
      probs_df = svc.predict(df)   # df: pandas DataFrame with original feature columns
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, schema_path: str = DEFAULT_SCHEMA):
        self.model_dir = model_dir
        self.schema_path = schema_path

        self.preproc = None
        self.feature_names: T.List[str] = []
        self.calibrators: T.Dict[str, T.Any] = {}
        self.boosters: T.Dict[str, T.Any] = {}

        self._load_artifacts()

    # --------------- Loading ---------------

    def _load_artifacts(self):
        # Preprocessor
        preproc_path = os.path.join(self.model_dir, "preprocessor.joblib")
        if not os.path.exists(preproc_path):
            raise FileNotFoundError(f"Missing preprocessor at {preproc_path}")
        self.preproc = joblib.load(preproc_path)

        # Feature schema
        if not os.path.exists(self.schema_path):
            # fallback to feature_names.txt
            txt = os.path.join(self.model_dir, "feature_names.txt")
            if os.path.exists(txt):
                with open(txt) as f:
                    self.feature_names = [l.strip() for l in f if l.strip()]
            else:
                raise FileNotFoundError(f"Missing schema at {self.schema_path} and fallback {txt}.")
        else:
            with open(self.schema_path) as f:
                obj = json.load(f)
                if isinstance(obj, dict):
                    # handle {"feature_names": [...]}
                    self.feature_names = obj.get("feature_names", [])
                else:
                    self.feature_names = list(obj)
        if not self.feature_names:
            raise RuntimeError("feature_names list is empty.")

        # Per-tag calibrators (preferred), otherwise boosters
        import xgboost as xgb  # lazy import here so module is present only when needed

        for tag in TAGS:
            cal_path = os.path.join(self.model_dir, f"calibrator_{tag}.joblib")
            if os.path.exists(cal_path):
                self.calibrators[tag] = joblib.load(cal_path)
            else:
                # Fallback to raw booster
                bst_path = os.path.join(self.model_dir, f"model_{tag}.json")
                if os.path.exists(bst_path):
                    booster = xgb.Booster()
                    booster.load_model(bst_path)
                    self.boosters[tag] = booster
                else:
                    warnings.warn(
                        f"Neither calibrator nor booster found for '{tag}'. "
                        f"Expected {cal_path} or {bst_path}. Predictions for this tag will be NaN."
                    )

    # --------------- Utilities ---------------

    def _ensure_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with exactly self.feature_names, adding any missing as zeros."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        # start from all-zero frame with correct columns & dtype float
        base = pd.DataFrame(0.0, index=X.index, columns=self.feature_names, dtype=float)
        # overlay any matching columns from input (type-cast to numeric where possible)
        common = [c for c in self.feature_names if c in X.columns]
        if common:
            # coerce to numeric where possible; non-numeric will be left as object and handled by preprocessor
            base[common] = X[common]
        return base[self.feature_names]  # correct order

    # --------------- Prediction ---------------

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pd.DataFrame
            Raw input features (any superset of self.feature_names).
            Index is preserved in the output.

        Returns
        -------
        pd.DataFrame
            Columns:
              - proba_visual_verbal
              - proba_active_reflective
              - proba_global_sequential
        """
        X_use = self._ensure_frame(X)
        # preprocess
        X_proc = self.preproc.transform(X_use)
        # predict per tag
        out = {}
        for tag in TAGS:
            if tag in self.calibrators:
                # CalibratedClassifierCV expects preprocessed array
                proba = self.calibrators[tag].predict_proba(X_proc)[:, 1]
            elif tag in self.boosters:
                # Booster expects DMatrix
                from xgboost import DMatrix
                proba = self.boosters[tag].predict(DMatrix(X_proc))
            else:
                proba = np.full(X_proc.shape[0], np.nan, dtype=float)
            out[f"proba_{tag}"] = proba.astype(float)

        return pd.DataFrame(out, index=X.index)

    # Convenience: single-row predict that returns a dict
    def predict_one(self, row: T.Union[pd.Series, dict]) -> dict:
        if isinstance(row, dict):
            row = pd.Series(row)
        df = row.to_frame().T
        df.index = [getattr(row, "name", 0)]
        res = self.predict(df)
        return res.iloc[0].to_dict()


# ---------------------- Optional FastAPI server ----------------------

if FASTAPI_AVAILABLE:
    app = FastAPI(title="Style Inference API", version="1.0.0")
    _SERVICE = None  # lazy-singleton

    class PredictRequest(BaseModel):
        # Accepts either a list of dicts or a single dict via "records"
        records: T.Union[T.List[dict], dict]

    class PredictResponse(BaseModel):
        results: T.List[dict]

    @app.on_event("startup")
    def _load_once():
        global _SERVICE
        if _SERVICE is None:
            model_dir = os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
            schema = os.environ.get("SCHEMA_PATH", DEFAULT_SCHEMA)
            _SERVICE = StyleInferenceService(model_dir=model_dir, schema_path=schema)

    @app.get("/health")
    def health():
        return {"status": "ok", "tags": list(TAGS)}

    @app.post("/predict", response_model=PredictResponse)
    def predict_api(req: PredictRequest):
        global _SERVICE
        if _SERVICE is None:
            _load_once()

        # normalize to list of dicts
        recs = req.records
        if isinstance(recs, dict):
            recs = [recs]
        X = pd.DataFrame(recs)
        preds = _SERVICE.predict(X)
        # return original records' index (positional) + probs
        results = []
        for i, (_, row) in enumerate(preds.iterrows()):
            results.append({
                "row_id": i,
                "proba_visual_verbal": float(row["proba_visual_verbal"]),
                "proba_active_reflective": float(row["proba_active_reflective"]),
                "proba_global_sequential": float(row["proba_global_sequential"]),
            })
        return {"results": results}

# --------------- CLI quick test ---------------
if __name__ == "__main__":
    # Minimal local smoke test (no FastAPI needed):
    svc = StyleInferenceService(model_dir=DEFAULT_MODEL_DIR, schema_path=DEFAULT_SCHEMA)
    # create a single zero row with correct columns
    dummy = pd.DataFrame([{}])
    res = svc.predict(dummy)
    print(res.head())
    print("OK")
