from PIL import Image
from ..models import Traits

"""
Extracts rough user traits from an uploaded image (very basic placeholder).

Purpose:
    Provides a minimal MVP implementation of trait extraction.
    It crops the center of the image, converts it to grayscale,
    and uses average brightness to estimate skin depth
    (light / medium / deep). Other traits are filled with
    default placeholder values for now.

Parameters:
    img (Image.Image):
        A PIL Image object uploaded by the user.

Process:
    1. Crop the central 40% of the image to reduce background noise.
    2. Convert the crop to grayscale and downsize to 32×32 for efficiency.
    3. Compute average brightness of the pixels.
    4. Map brightness value to one of three skin depth categories.
    5. Populate a Traits object with this skin depth and default values
       for other attributes.

Returns:
    Traits:
        An object containing:
        - skin_temperature: fixed "neutral" (placeholder)
        - skin_depth: "light" / "medium" / "deep" (based on brightness)
        - hair_type, hair_color: fixed "unknown"
        - frame: "regular"
        - height_bucket: "avg"
        - shoulders: "average"
"""
def quick_traits_from_image(img: Image.Image) -> Traits:
    # Get the image width and height
    w, h = img.size

    # Define crop coordinates for the central region (30% → 70% both ways)
    cx1, cy1 = int(w * 0.3), int(h * 0.3)
    cx2, cy2 = int(w * 0.7), int(h * 0.7)

    # Crop the central region, convert to grayscale, and resize to 32×32
    crop = img.crop((cx1, cy1, cx2, cy2)).convert("L").resize((32, 32))

    # Compute average brightness of the cropped grayscale image
    mean = sum(crop.getdata()) / (32 * 32)

    # Map brightness to a rough skin depth category
    if mean > 180:
        skin_depth = "light"
    elif mean > 110:
        skin_depth = "medium"
    else:
        skin_depth = "deep"

    # Build a Traits object with guessed skin_depth
    # Other attributes are fixed/default placeholders for now
    traits = Traits(
        skin_temperature="neutral",  # fixed value for MVP
        skin_depth=skin_depth,       # derived from brightness
        hair_type="unknown",
        hair_color="unknown",
        frame="regular",
        height_bucket="avg",
        shoulders="average",
    )

    # Return the traits object
    return traits