# ------------------------------------------------------------
# Helpers for ML scoring:
#   - load_pipe(): load & cache the trained pipeline
#   - row_from(): build a feature row dict for model input
# ------------------------------------------------------------

from pathlib import Path
from typing import Dict, Any, Optional
import joblib
from ...models import Traits, Style

# Paths
ML_DIR = Path(__file__).resolve().parents[1]       # .../src/app/ml
MODEL_PATH = ML_DIR / "reco_lr.joblib"             # saved pipeline path

# Cache
_PIPE = None

"""
    Load the trained pipeline from disk once and cache it.

    Raises:
        FileNotFoundError: if the model file doesn't exist.

    Returns:
        sklearn pipeline object
    """

def load_pipe():
    global _PIPE
    if _PIPE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train it with: python -m src.app.ml.train_model"
            )
        _PIPE = joblib.load(MODEL_PATH)
    return _PIPE


"""
    Build one feature row dict for ML model input.
    """
def row_from(prod: Dict[str, Any], traits: Traits, style: Style, size: Optional[str]) -> dict:
    
    # Price (default 0 if missing)
    price = int(prod.get("price") or 0)

    # Size availability flag
    sizes = prod.get("sizes") or []
    has_size = int(bool(size and size in sizes))

    # Tags normalized to semicolon string
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
