from typing import List, Dict, Any, Tuple, Optional
from .rules import palette_for, fit_tags_for, avoid_tags_for
from .models import Traits, Style


def score_product(
    prod: Dict[str, Any],
    traits: Traits,
    style: Style,
    size: Optional[str],
    budget: Optional[int],
) -> Tuple[float, List[str]]:
    why: List[str] = []

    # Normalize tags to lowercase
    tags: List[str] = [str(t).lower() for t in (prod.get("tags") or [])]

    color_hit = 0.0
    fit_hit = 0.0
    diversity = 0.1  # light constant to avoid ties

    # Color match
    wanted_colors = [c.lower() for c in palette_for(traits)]
    if any(c in tags for c in wanted_colors):
        color_hit = 1.0
        why.append("matches your color palette")

    # Fit / neckline / length
    wanted_tags = [t.lower() for t in fit_tags_for(traits, style)]
    if any(t in tags for t in wanted_tags):
        fit_hit = 1.0
        why.append("fit/neckline suits your build")

    # Avoid tags
    bad = [t.lower() for t in avoid_tags_for(traits, style)]
    if any(t in tags for t in bad):
        return -1.0, ["not ideal for your build"]

    # Size / stock
    sizes = prod.get("sizes") or []
    if size and size not in sizes:
        return -0.5, ["size unavailable"]
    if size:
        why.append("in stock in your size")

    # Price alignment
    price = int(prod.get("price") or 0)
    price_score = 0.6  # neutral baseline if no budget
    if budget is not None and budget > 0:
        if price <= budget:
            price_score = 1.0
            why.append(f"within budget (₹{price})")
        else:
            # Penalize the farther it is above budget (clamped)
            over = max(0, price - budget)
            price_score = max(0.2, 1.0 - over / budget)
            if price_score > 0.6:
                why.append("near your budget")

    # Final weighted score
    score = 0.45 * color_hit + 0.35 * fit_hit + 0.15 * price_score + 0.05 * diversity
    return score, (why if why else ["good overall match"])
