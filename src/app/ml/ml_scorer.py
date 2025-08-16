# src/app/ml/ml_scorer.py
# ------------------------------------------------------------
# Uses the saved training pipeline (preprocessing + model)
# to score products for a user's traits.
# ------------------------------------------------------------

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import pandas as pd  # ← IMPORTANT: ensure we pass a DataFrame to the pipeline
from .utils.ml_scorer_utils import load_pipe
ML_DIR = Path(__file__).resolve().parent               # .../src/app/ml
MODEL_PATH = ML_DIR / "reco_lr.joblib"                 # saved pipeline path

"""
Scores multiple products at once using the ML pipeline.

Purpose:
    Loads the trained pipeline, converts a list of feature rows into
    a DataFrame, and returns the predicted probabilities for each row.

Parameters:
    rows (List[dict]):
        A list of feature dictionaries, each matching the model’s
        expected columns (as built by `row_from`).

Returns:
    List[float]:
        A list of predicted probabilities (0–1), one per row.
        If no rows are provided, returns an empty list.
"""
def ml_predict_probas(rows: List[dict]) -> List[float]:
    pipe = load_pipe()
    if not rows:
        return []
    df = pd.DataFrame(rows)
    return pipe.predict_proba(df)[:, 1].tolist()
