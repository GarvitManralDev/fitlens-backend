# src/app/ml/synthetic_data_generator.py
from pathlib import Path
import csv, random, uuid

ML_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = ML_DIR / "events_training_data.csv"

def generate_csv(path: Path = DEFAULT_CSV, n_sessions: int = 200, items_per_session: int = 16):
    """
    Create a synthetic dataset in ml/events_training_data.csv (by default)
    so you can train immediately. Matches train_model.py's expected schema.
    """
    styles = ["casual", "traditional"]
    skin_temps = ["warm", "cool", "neutral"]
    skin_depths = ["light", "medium", "deep"]
    frames = ["slim", "regular", "fuller"]
    heights = ["short", "avg", "tall"]
    shoulders = ["narrow", "average", "broad"]

    color_vocab = {
        "warm": ["olive","mustard","rust","warm beige","cream","maroon","brown","tan"],
        "cool": ["navy","charcoal","cool gray","emerald","wine","ice blue","black","white"],
        "neutral": ["taupe","sand","stone","black","white","navy","olive"]
    }
    fit_tags_map = {
        "slim": ["slim","regular","structured-shoulder"],
        "regular": ["regular"],
        "fuller": ["relaxed","straight","drape","no-cling"],
    }
    height_tags_map = {"short": ["regular-length","cropped"], "avg": [], "tall": ["longline","layer-friendly"]}
    shoulder_tags_map = {"narrow": ["mandarin","crew","stand-collar"], "average": [], "broad": ["v-neck","henley","short-mandarin"]}

    def sample_traits():
        return {
            "style": random.choice(styles),
            "skin_temperature": random.choice(skin_temps),
            "skin_depth": random.choice(skin_depths),
            "frame": random.choice(frames),
            "height_bucket": random.choice(heights),
            "shoulders": random.choice(shoulders),
            "budget": random.choice([799, 999, 1299, 1499, 1999, 2499, 2999]),
            "size": random.choice([None, "S","M","L","XL"]),
        }

    def sample_product(style):
        pid = "p_" + uuid.uuid4().hex[:8]
        base_price = random.choice([599,799,999,1299,1499,1799,1999,2499,2999,3499])
        sizes = random.sample(["S","M","L","XL"], k=random.randint(1,4))
        common = ["casual"] if style=="casual" else ["traditional"]
        random_colors = random.sample(
            ["olive","mustard","rust","warm beige","cream","maroon","brown","tan",
             "navy","charcoal","cool gray","emerald","wine","ice blue","black","white",
             "taupe","sand","stone"], k=random.randint(1,3)
        )
        random_fit = random.sample(["slim","regular","structured-shoulder","relaxed","straight","drape","no-cling"], k=random.randint(1,2))
        random_neck = random.sample(["mandarin","crew","stand-collar","v-neck","henley","short-mandarin"], k=random.randint(0,1))
        tags = list(set(common + random_colors + random_fit + random_neck))
        return {"product_id": pid, "price": base_price, "sizes": sizes, "tags": tags}

    def label_example(tr, prod):
        score = 0.0
        wanted_colors = [c.lower() for c in color_vocab[tr["skin_temperature"]]]
        if any(c in [t.lower() for t in prod["tags"]] for c in wanted_colors): score += 0.8
        if any(t in fit_tags_map[tr["frame"]] for t in prod["tags"]): score += 0.6
        if any(t in height_tags_map[tr["height_bucket"]] for t in prod["tags"]): score += 0.25
        if any(t in shoulder_tags_map[tr["shoulders"]] for t in prod["tags"]): score += 0.25
        has_size = int(bool(tr["size"] and tr["size"] in prod["sizes"]))
        if has_size: score += 0.5
        if prod["price"] <= tr["budget"]: score += 0.4
        else:
            over = (prod["price"] - tr["budget"]) / max(1, tr["budget"])
            score -= min(0.6, over)
        score += random.uniform(-0.2, 0.2)
        prob = max(0.0, min(1.0, 0.15 + 0.2*score))
        return 1 if random.random() < prob else 0, has_size

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "session_id","slate_id","product_id","label",
            "price","has_size","style","skin_temperature","skin_depth","frame","height_bucket","shoulders",
            "color_tags","fit_tags","avoid_tags","rank_in_slate"
        ])
        for _ in range(n_sessions):
            session_id = "s_" + uuid.uuid4().hex[:8]
            slate_id = "sl_" + uuid.uuid4().hex[:8]
            tr = sample_traits()
            products = [sample_product(tr["style"]) for _ in range(items_per_session)]
            random.shuffle(products)
            for rank, prod in enumerate(products):
                y, has_size = label_example(tr, prod)
                tags_join = ";".join([t for t in prod["tags"] if t])
                w.writerow([
                    session_id, slate_id, prod["product_id"], y,
                    prod["price"], has_size, tr["style"], tr["skin_temperature"], tr["skin_depth"],
                    tr["frame"], tr["height_bucket"], tr["shoulders"],
                    tags_join, tags_join, "", rank
                ])
    print(f"[synthetic_data_generator] Wrote → {path}")

if __name__ == "__main__":
    generate_csv()  # writes ml/events_training_data.csv
