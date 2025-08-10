# src/app/ml/ml_scorer.py
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import joblib
from ..models import Traits, Style

ML_DIR = Path(__file__).resolve().parent               # .../src/app/ml
MODEL_PATH = ML_DIR / "reco_lr.joblib"                 # load model from ml/

_PIPE = None

def _load_pipe():
    global _PIPE
    if _PIPE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train it with: python -m src.app.ml.train_model"
            )
        _PIPE = joblib.load(MODEL_PATH)
    return _PIPE

def _row_from(prod: Dict[str, Any], traits: Traits, style: Style, size: Optional[str]) -> dict:
    price = int(prod.get("price") or 0)
    sizes = prod.get("sizes") or []
    has_size = int(bool(size and size in sizes))

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
    pipe = _load_pipe()
    row = _row_from(prod, traits, style, size)
    proba = float(pipe.predict_proba([row])[0, 1])

    why: List[str] = []
    if size and size in (prod.get("sizes") or []):
        why.append("in stock in your size")
    if prod.get("price") is not None:
        why.append(f"price ₹{int(prod['price'])}")

    return proba, (why or ["good predicted match"])
