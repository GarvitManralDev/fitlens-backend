from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# Restrict style to these two strings (gives nice validation + docs)
Style = Literal["casual", "traditional"]


class Traits(BaseModel):
    """
    Minimal trait set used by the recommender.
    (Your image analyzer fills these; feel free to expand later.)
    """
    skin_temperature: Literal["warm", "cool", "neutral"]
    skin_depth: Literal["light", "medium", "deep"]
    hair_type: Literal["straight", "wavy", "curly", "unknown"] = "unknown"
    hair_color: Literal["dark", "brown", "blonde", "other", "unknown"] = "unknown"
    frame: Literal["slim", "regular", "fuller"]
    height_bucket: Literal["short", "avg", "tall"]
    shoulders: Literal["narrow", "broad", "average"]


class ProductOut(BaseModel):
    """
    What we return for each recommended product.
    """
    id: str
    title: str
    store: str
    url: str
    image: str
    price: int
    mrp: Optional[int] = None
    # Use empty lists by default (cleaner than None checks downstream)
    sizes: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    # Short, human-friendly reasons for why this product was recommended
    why: List[str]


class RecommendResponse(BaseModel):
    """
    Top-level response containing a list of products.
    """
    items: List[ProductOut]


class TrackEvent(BaseModel):
    """
    Event we log from the client for training data later.
    """
    event: Literal["click", "like", "hide"]
    product_id: str
    session_id: str
