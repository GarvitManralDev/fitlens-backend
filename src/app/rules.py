from typing import List
from .models import Traits, Style

# Small, explicit rule tables for MVP

WARM_COLORS = ["olive", "mustard", "rust", "warm beige", "cream", "maroon"]
COOL_COLORS = ["navy", "charcoal", "cool gray", "emerald", "wine", "ice blue"]
NEUTRAL_COLORS = ["taupe", "sand", "stone", "black", "white"]


def palette_for(traits: Traits) -> List[str]:
    """Return preferred color palette from skin undertone."""
    if traits.skin_temperature == "warm":
        return WARM_COLORS + ["brown", "tan"]
    if traits.skin_temperature == "cool":
        return COOL_COLORS + ["black", "white"]
    return NEUTRAL_COLORS + ["navy", "olive"]


def fit_tags_for(traits: Traits, style: Style) -> List[str]:
    """Return positive tags (fit/neckline/length + category) based on traits."""
    tags: List[str] = []

    # Frame → fit
    if traits.frame == "slim":
        tags += ["slim", "regular", "structured-shoulder"]
    elif traits.frame == "regular":
        tags += ["regular"]
    else:  # fuller
        tags += ["relaxed", "straight", "drape", "no-cling"]

    # Height → length
    if traits.height_bucket == "short":
        tags += ["regular-length", "cropped"]
    elif traits.height_bucket == "tall":
        tags += ["longline", "layer-friendly"]

    # Shoulders → neckline/collar
    if traits.shoulders == "narrow":
        tags += ["mandarin", "crew", "stand-collar"]
    elif traits.shoulders == "broad":
        tags += ["v-neck", "henley", "short-mandarin"]

    # Category
    tags += ["casual"] if style == "casual" else ["traditional"]

    return tags


def avoid_tags_for(traits: Traits, style: Style) -> List[str]:
    """Return tags to avoid for this body type."""
    avoid: List[str] = []
    if traits.frame == "fuller":
        avoid += ["clingy", "heavy-shine"]
    if traits.height_bucket == "short":
        avoid += ["extra-long"]
    return avoid


def category_allowed(style: Style) -> List[str]:
    """Return allowed product categories for current style."""
    return ["casual"] if style == "casual" else ["traditional"]
