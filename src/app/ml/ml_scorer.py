# src/app/ml/ml_scorer.py
# ------------------------------------------------------------
# Uses the saved training pipeline (preprocessing + model)
# to score products for a user's traits.
# ------------------------------------------------------------
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import joblib
import pandas as pd  # ← IMPORTANT: ensure we pass a DataFrame to the pipeline
from ..models import Traits, Style

ML_DIR = Path(__file__).resolve().parent               # .../src/app/ml
MODEL_PATH = ML_DIR / "reco_lr.joblib"                 # saved pipeline path

_PIPE = None  # cache the loaded pipeline so we don't reload every call


def _load_pipe():
    """
    Steps 1–2 (from your logic notes):
    - Check the model file exists.
    - Load it once and cache it for reuse.
    """
    global _PIPE
    if _PIPE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train it with: python -m src.app.ml.train_model"
            )
        _PIPE = joblib.load(MODEL_PATH)
    return _PIPE


def _row_from(prod: Dict[str, Any], traits: Traits, style: Style, size: Optional[str]) -> dict:
    """
    Steps 3–6:
    Build ONE feature row with the exact columns used during training.
    """
    price = int(prod.get("price") or 0)

    sizes = prod.get("sizes") or []
    has_size = int(bool(size and size in sizes))

    # Join product tags into a single semicolon-separated string
    tags = [str(t).lower() for t in (prod.get("tags") or [])]
    semis = ";".join(t for t in tags if t)

    return {
        "price": price,
        "has_size": has_size,
        "style": style,
        "skin_temperature": traits.skin_temperature,
        "skin_depth": traits.skin_depth,
        "frame": traits.frame,
        "height_bucket": traits.height_bucket,
        "shoulders": traits.shoulders,
        "color_tags": semis,
        "fit_tags": semis,
        "avoid_tags": "",
    }


def ml_score_product(
    prod: Dict[str, Any],
    traits: Traits,
    style: Style,
    size: Optional[str],
) -> Tuple[float, List[str]]:
    """
    Steps 7–9:
    - Create a one-row DataFrame (required so ColumnTransformer can select by column names).
    - Get probability of label=1 (user will like it).
    - Generate simple 'why' reasons.
    """
    pipe = _load_pipe()

    row = _row_from(prod, traits, style, size)
    df = pd.DataFrame([row])  # ← CRITICAL FIX: pass a DataFrame, not a list of dicts
    proba = float(pipe.predict_proba(df)[0, 1])

    why: List[str] = []
    if size and size in (prod.get("sizes") or []):
        why.append("in stock in your size")
    if prod.get("price") is not None:
        why.append(f"price ₹{int(prod['price'])}")

    return proba, (why or ["good predicted match"])


def ml_predict_probas(rows: List[dict]) -> List[float]:
    """
    Optional helper for batching:
    Accept a list of *already-built* feature rows (same schema as _row_from),
    return a list of probabilities for label=1.
    """
    pipe = _load_pipe()
    if not rows:
        return []
    df = pd.DataFrame(rows)
    return pipe.predict_proba(df)[:, 1].tolist()
