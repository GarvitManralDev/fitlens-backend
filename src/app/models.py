from typing import List, Literal, Optional
from pydantic import BaseModel

Style = Literal["casual", "traditional"]


class Traits(BaseModel):
    # minimal trait set for MVP; expand later if needed
    skin_temperature: Literal["warm", "cool", "neutral"]
    skin_depth: Literal["light", "medium", "deep"]
    hair_type: Literal["straight", "wavy", "curly", "unknown"] = "unknown"
    hair_color: Literal["dark", "brown", "blonde", "other", "unknown"] = "unknown"
    frame: Literal["slim", "regular", "fuller"]
    height_bucket: Literal["short", "avg", "tall"]
    shoulders: Literal["narrow", "broad", "average"]


class ProductOut(BaseModel):
    id: str
    title: str
    store: str
    url: str
    image: str
    price: int
    mrp: Optional[int] = None
    sizes: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    why: List[str]


class RecommendResponse(BaseModel):
    items: List[ProductOut]


class TrackEvent(BaseModel):
    event: Literal["click", "like", "hide"]
    product_id: str
    session_id: str
