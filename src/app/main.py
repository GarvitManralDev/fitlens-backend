from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from PIL import Image
import io, time

from .db import get_client
from .models import RecommendResponse, ProductOut, TrackEvent, Traits, Style

# --- ML scorer ONLY (no rule-based fallback) ---
from .ml.ml_scorer import ml_score_product  # (prod, traits, style, size) -> (score, why)

# -----------------------------
# FastAPI app configuration
# -----------------------------
app = FastAPI(title="FitLens Backend (ML-Only)")

# CORS is open for MVP; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # TODO: replace with your web app's URL
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Helper: ultra-simple trait extraction from image (MVP placeholder)
# NOTE: This is NOT ML; just a stub until you replace with a real model.
# -----------------------------
def quick_traits_from_image(img: Image.Image) -> Traits:
    """
    Very basic placeholder:
    - Crops the center of the image
    - Converts to grayscale
    - Uses average brightness to guess skin depth (light/medium/deep)
    Other traits are default constants for now.
    """
    w, h = img.size
    cx1, cy1 = int(w * 0.3), int(h * 0.3)
    cx2, cy2 = int(w * 0.7), int(h * 0.7)
    crop = img.crop((cx1, cy1, cx2, cy2)).convert("L").resize((32, 32))
    mean = sum(crop.getdata()) / (32 * 32)

    if mean > 180:
        skin_depth = "light"
    elif mean > 110:
        skin_depth = "medium"
    else:
        skin_depth = "deep"

    traits = Traits(
        skin_temperature="neutral",  # fixed for now; upgrade later
        skin_depth=skin_depth,
        hair_type="unknown",
        hair_color="unknown",
        frame="regular",
        height_bucket="avg",
        shoulders="average",
    )
    return traits


# -----------------------------
# Endpoint: analyze image → fetch products → ML-score → return top picks
# -----------------------------
@app.post("/analyze-and-recommend", response_model=RecommendResponse)
async def analyze_and_recommend(
    image: UploadFile = File(...),          # User photo upload
    style: Style = Form(...),               # "casual" or "traditional" (Pydantic Literal)
    size: Optional[str] = Form(default=None),     # Optional user size (e.g., "M" or "32"); used as a feature
    budget: Optional[int] = Form(default=None),   # Currently NOT used here; keep for future training features
):
    """
    ML-only pipeline:
      1) Validate + load image
      2) Extract rough Traits (MVP stub)
      3) Fetch ALL product candidates from DB (no rule-based category gating)
      4) Join price/sizes (features needed by the model)
      5) Minimal filtering (drop out-of-stock / missing price only)
      6) Score via ML model (no rule heuristics), sort by score
      7) Return top 12
    """

    # --- 1) Validate + load image ---
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
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

    price_map: Dict[str, Dict[str, Any]] = {}
    for row in (price_res.data or []):
        price_map[row["product_id"]] = row

    enriched: List[Dict[str, Any]] = []
    for p in products:
        pr = price_map.get(p["id"])
        if not pr:
            continue  # skip products without price (model needs price feature)
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
    # We do NOT filter by size or budget (ML handles "has_size" and "price" as features).
    # We only drop items not in stock (UX reason) or with missing price (feature missing).
    filtered: List[Dict[str, Any]] = []
    for p in enriched:
        if not p.get("in_stock", True):
            continue
        if p.get("price") is None:
            continue
        filtered.append(p)

    # --- 6) ML score & sort ---
    scored: List[tuple[float, Dict[str, Any], List[str]]] = []
    for p in filtered:
        s, why = ml_score_product(p, traits, style, size)
        if s > 0:
            scored.append((float(s), p, why))

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


# -----------------------------
# Endpoint: track engagement events (click/like, etc.)
# -----------------------------
@app.post("/track")
async def track(event: TrackEvent):
    """
    Record a simple user interaction with a product.
    (This data is what you'll later export to train better models.)
    """
    sb = get_client()
    table = "clicks" if event.event == "click" else "likes" if event.event == "like" else "likes"
    res = sb.table(table).insert(
        {
            "product_id": event.product_id,
            "session_id": event.session_id,
            "ts": int(time.time()),
        }
    ).execute()
    if getattr(res, "error", None):
        raise HTTPException(500, res.error.message)
    return {"ok": True}
