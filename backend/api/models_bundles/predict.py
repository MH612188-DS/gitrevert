# /content/outputs/predict.py
import os, json, joblib, numpy as np, pandas as pd

MODEL_DIR = "/content/outputs"
SCHEMA_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

TAGS = [
    ("visual_verbal",     "proba_visual_verbal",     "pred_visual_verbal"),
    ("active_reflective", "proba_active_reflective", "pred_active_reflective"),
    ("global_sequential", "proba_global_sequential", "pred_global_sequential"),
]

class StyleInferenceService:
    def __init__(self, model_dir: str = MODEL_DIR, schema_path: str = SCHEMA_PATH):
        self.model_dir = model_dir
        self.schema_path = schema_path
        self.expected_cols = self._load_schema()
        self.calibrators = self._load_models()

    # ---------- PUBLIC ----------
    def predict(self, df_features: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        df_features: engineered numeric features with index = id_student.
                     Columns can be a superset/subset; function will align to training schema.
        Returns: DataFrame with calibrated probs, class labels, and confidence per dimension.
        """
        X = self._align_features(df_features)
        out = pd.DataFrame(index=X.index)

        for tag, proba_col, pred_col in TAGS:
            calib = self.calibrators[tag]
            p = calib.predict_proba(X)[:, 1]
            yhat = (p >= threshold).astype(int)
            out[proba_col] = p
            out[pred_col]  = yhat
            out[f"conf_{tag}"] = np.abs(p - 0.5) * 2.0  # 0..1 distance from 0.5

        return out

    # ---------- INTERNALS ----------
    def _load_schema(self):
        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        with open(self.schema_path, "r") as f:
            cols = json.load(f)
        if not isinstance(cols, list) or not cols:
            raise ValueError("feature_columns.json is invalid or empty.")
        return cols

    def _load_models(self):
        calibs = {}
        for tag, _, _ in TAGS:
            path = os.path.join(self.model_dir, f"calibrator_{tag}.joblib")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing calibrator: {path}")
            calibs[tag] = joblib.load(path)  # CalibratedClassifierCV(prefit)
        return calibs

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # ensure index name
        if df.index.name is None:
            df.index.name = "id_student"

        # add any missing columns as 0
        missing = [c for c in self.expected_cols if c not in df.columns]
        if missing:
            df = df.copy()
            for c in missing:
                df[c] = 0.0

        # drop unexpected extras
        extras = [c for c in df.columns if c not in self.expected_cols]
        if extras:
            df = df.drop(columns=extras)

        # reorder and enforce numeric dtype
        X = df[self.expected_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        # float32 is plenty for GBMs
        return X.astype(np.float32)
