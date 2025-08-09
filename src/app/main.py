from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from PIL import Image
import io, time

from .db import get_client
from .models import RecommendResponse, ProductOut, TrackEvent, Traits
from .rules import category_allowed
from .scoring import score_product

app = FastAPI(title="FitLens Backend (MVP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helpers ---

def quick_traits_from_image(img: Image.Image) -> Traits:
    """
    MVP placeholder: derive very rough traits from image brightness/contrast.
    Replace with real landmarks/pose later.
    """
    # Downsample center crop
    w, h = img.size
    cx1, cy1 = int(w * 0.3), int(h * 0.3)
    cx2, cy2 = int(w * 0.7), int(h * 0.7)
    crop = img.crop((cx1, cy1, cx2, cy2)).convert("L").resize((32, 32))
    mean = sum(crop.getdata()) / (32 * 32)

    # naive buckets
    skin_depth = "light" if mean > 180 else "medium" if mean > 110 else "deep"
    skin_temperature = "neutral"  # keep neutral until you add real color clustering

    traits = Traits(
        skin_temperature=skin_temperature,
        skin_depth=skin_depth,
        hair_type="unknown",
        hair_color="unknown",
        frame="regular",
        height_bucket="avg",
        shoulders="average",
    )
    return traits


# --- Endpoints ---

@app.post("/analyze-and-recommend", response_model=RecommendResponse)
async def analyze_and_recommend(
    image: UploadFile = File(...),
    style: str = Form(..., pattern="^(casual|traditional)$"),
    size: Optional[str] = Form(default=None),
    budget: Optional[int] = Form(default=None),
):
    # 1) read image fully in memory, basic checks
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(400, "Unsupported image type")
    data = await image.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    # 2) compute traits (MVP placeholder)
    traits = quick_traits_from_image(img)

    # 3) fetch candidates from Supabase
    sb = get_client()
    allowed = category_allowed(style)  # ["casual"] or ["traditional"]

    prod_res = sb.table("products").select("*").in_("category", allowed).execute()
    if prod_res.error:
        raise HTTPException(500, prod_res.error.message)
    products: List[Dict[str, Any]] = prod_res.data or []

    # fetch prices for all product ids
    ids = [p["id"] for p in products] or ["-"]
    price_res = sb.table("prices").select("*").in_("product_id", ids).execute()
    price_map: Dict[str, Dict[str, Any]] = {}
    for row in (price_res.data or []):
        price_map[row["product_id"]] = row

    # Attach price & sizes into each product dict
    enriched: List[Dict[str, Any]] = []
    for p in products:
        pr = price_map.get(p["id"])
        if not pr:
            continue
        enriched.append(
            {
                **p,
                "price": pr.get("price"),
                "mrp": pr.get("mrp"),
                "sizes": pr.get("sizes") or [],
                "in_stock": pr.get("in_stock", True),
            }
        )

    # 4) filter quick (size, stock, budget gate)
    filtered: List[Dict[str, Any]] = []
    for p in enriched:
        if not p.get("in_stock", True):
            continue
        if size and size not in (p.get("sizes") or []):
            continue
        if budget and p.get("price") and p["price"] > budget * 1.7:  # loose cap
            continue
        filtered.append(p)

    # 5) score & sort
    scored: List[tuple[float, Dict[str, Any], list[str]]] = []
    for p in filtered:
        s, why = score_product(p, traits, style, size, budget)
        if s > 0:
            scored.append((s, p, why))
    scored.sort(key=lambda t: t[0], reverse=True)

    top = []
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


@app.post("/track")
async def track(event: TrackEvent):
    sb = get_client()
    table = "clicks" if event.event == "click" else "likes" if event.event == "like" else "likes"
    # For "hide", create a separate table later if needed.
    res = sb.table(table).insert(
        {"product_id": event.product_id, "session_id": event.session_id, "ts": int(time.time())}
    ).execute()
    if res.error:
        raise HTTPException(500, res.error.message)
    return {"ok": True}
