from fastapi import UploadFile, File, Form, HTTPException
from typing import Optional, List, Dict, Any
from PIL import Image
import io, time
from fastapi import APIRouter
from ..db import get_client
from ..models import RecommendResponse, ProductOut, TrackEvent, Style
from ..ml.ml_scorer import ml_predict_probas  
from ..ml.utils.ml_scorer_utils import row_from
from ..utils.analyze_utils import quick_traits_from_image

router = APIRouter(tags=["recommendations"])

"""
End-to-end endpoint: Analyze user image and return top outfit recommendations.

Purpose:
    This FastAPI route accepts a user photo and style choice, extracts
    traits from the image, enriches product data with prices/sizes, and
    runs the ML model to score all products. It then sorts the products
    by predicted match probability and returns the top results.

Parameters:
    image (UploadFile):
        User-uploaded image (JPEG/PNG/WebP). Used to extract rough traits.
    style (Style):
        Desired outfit style (Literal: "casual" or "traditional").
    size (Optional[str]):
        User’s preferred size. Used to compute the 'has_size' feature and
        provide "in stock in your size" explanations.
    budget (Optional[int]):
        Reserved for future use (not applied in the MVP).

Process:
    1. Validate + load image file.
    2. Extract rough traits from image (stub function in MVP).
    3. Fetch all products from DB (no category gating).
    4. Join products with price/size info from 'prices' table.
    5. Apply minimal filtering (drop items without price or out of stock).
    6. Build feature rows, run batch ML predictions, and generate reasons.
    7. Sort products by score and return the top 12.

Returns:
    RecommendResponse:
        A structured response containing up to 12 top product
        recommendations, each with product metadata, price/MRP,
        available sizes, tags, and explanation reasons.
"""
@router.post("/analyze-and-recommend", response_model=RecommendResponse)
async def analyze_and_recommend(
    image: UploadFile = File(...),                # User photo upload
    style: Style = Form(...),                     # "casual" or "traditional" (Pydantic Literal)
    size: Optional[str] = Form(default=None),     # Optional user size; becomes 'has_size' feature
    budget: Optional[int] = Form(default=None),   # Not used in this MVP (reserved for future features)
):

    # --- 1) Validate + load image ---
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "Unsupported image type")
    data = await image.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    # --- 2) Extract Traits (stub) ---
    traits = quick_traits_from_image(img)

    # --- 3) Fetch ALL products (no style/category gating) ---
    sb = get_client()
    prod_res = sb.table("products").select("*").execute()
    if getattr(prod_res, "error", None):
        raise HTTPException(500, prod_res.error.message)
    products: List[Dict[str, Any]] = prod_res.data or []

    # --- 4) Join price/sizes for each product ---
    ids = [p["id"] for p in products] or ["-"]  # avoid empty IN() calls
    price_res = sb.table("prices").select("*").in_("product_id", ids).execute()
    if getattr(price_res, "error", None):
        raise HTTPException(500, price_res.error.message)

    price_map: Dict[str, Dict[str, Any]] = {row["product_id"]: row for row in (price_res.data or [])}

    enriched: List[Dict[str, Any]] = []
    for p in products:
        pr = price_map.get(p["id"])
        if not pr:
            continue  # skip products without price (model needs 'price')
        enriched.append(
            {
                **p,
                "price": pr.get("price"),
                "mrp": pr.get("mrp"),
                "sizes": pr.get("sizes") or [],
                "in_stock": pr.get("in_stock", True),
            }
        )

    # --- 5) Minimal filtering only ---
    # Do NOT filter by size/budget here; the model consumes 'has_size' and 'price'.
    filtered: List[Dict[str, Any]] = [
        p for p in enriched
        if p.get("in_stock", True) and (p.get("price") is not None)
    ]

    if not filtered:
        return RecommendResponse(items=[])

    # --- 6) ML score (BATCH) & sort ---
    # Build feature rows *once*, predict probabilities in one call (faster than per-item loop).
    rows: List[dict] = [row_from(p, traits, style, size) for p in filtered]
    probas: List[float] = ml_predict_probas(rows)

    # Build simple 'why' reasons alongside scores
    scored: List[tuple[float, Dict[str, Any], List[str]]] = []
    for p, s in zip(filtered, probas):
        why: List[str] = []
        if size and size in (p.get("sizes") or []):
            why.append("in stock in your size")
        if p.get("price") is not None:
            why.append(f"price ₹{int(p['price'])}")
        scored.append((float(s), p, (why or ["good predicted match"])))

    scored.sort(key=lambda t: t[0], reverse=True)

    # --- 7) Build response (top 12) ---
    top: List[ProductOut] = []
    for _, p, why in scored[:12]:
        top.append(
            ProductOut(
                id=p["id"],
                title=p["title"],
                store=p["store"],
                url=p["url"],
                image=p["image"],
                price=int(p["price"]) if p.get("price") is not None else 0,
                mrp=int(p["mrp"]) if p.get("mrp") is not None else None,
                sizes=p.get("sizes"),
                tags=p.get("tags"),
                why=why,
            )
        )

    return RecommendResponse(items=top)


"""
Track user interactions with products (clicks / likes).

Purpose:
    Records a user’s interaction event (click or like) into the
    appropriate database table via Supabase. Each record stores
    the product ID, session ID, and a timestamp.

Parameters:
    event (TrackEvent):
        A Pydantic model containing:
          - event: type of interaction ("click" or "like")
          - product_id: the product being interacted with
          - session_id: unique session identifier for the user

Process:
    1. Determine target table ("clicks" for clicks, "likes" for likes).
    2. Insert a new row with product_id, session_id, and current timestamp.
    3. Handle DB errors gracefully and raise 500 if insertion fails.

Returns:
    dict:
        { "ok": True } if the insert was successful.
"""
@router.post("/track")
async def track(event: TrackEvent):
    # Get a Supabase client instance
    sb = get_client()

    # Choose the target table based on the event type
    # → "clicks" for click events, "likes" for like events (default to "likes")
    table = "clicks" if event.event == "click" else "likes" if event.event == "like" else "likes"

    # Insert a new row into the chosen table with product_id, session_id, and timestamp
    res = sb.table(table).insert(
        {
            "product_id": event.product_id,
            "session_id": event.session_id,
            "ts": int(time.time()),   # current Unix timestamp
        }
    ).execute()

    # If the DB operation returned an error → raise 500
    if getattr(res, "error", None):
        raise HTTPException(500, res.error.message)

    # Return success response if insert worked
    return {"ok": True}