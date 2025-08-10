# src/app/ml/train_model.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import joblib

ML_DIR = Path(__file__).resolve().parent
INPUT_CSV = ML_DIR / "events_training_data.csv"
MODEL_PATH = ML_DIR / "reco_lr.joblib"

# ---- picklable helpers ----
def prep_text_col(X):
    """(n,1) -> (n,) strings; replace blanks with '__none__' so vocab isn't empty."""
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    arr = np.ravel(X).astype(str)
    # normalize empties
    bad = (arr == "") | (arr == "nan") | (arr == "None") | (arr == "NONE")
    arr[bad] = "__none__"
    return arr

def split_semicolon(s: str):
    """Tokenizer for semicolon-separated tags; keeps '__none__' intact."""
    if s is None:
        return []
    s = str(s).strip().lower()
    if s == "":
        return ["__none__"]
    # if it was our placeholder, keep it as a single token
    if s == "__none__":
        return ["__none__"]
    return [tok.strip() for tok in s.split(";") if tok.strip()]

def main():
    # If CSV missing, auto-generate synthetic one inside ml/
    if not INPUT_CSV.exists():
        from .synthetic_data_generator import generate_csv
        print(f"[train_model] CSV not found. Generating at: {INPUT_CSV}")
        generate_csv(INPUT_CSV, n_sessions=200, items_per_session=16)

    df = pd.read_csv(INPUT_CSV)

    if "label" not in df.columns:
        raise ValueError("CSV must include a 'label' column with 1/0.")

    # Numeric
    df["price"] = pd.to_numeric(df.get("price", 0), errors="coerce").fillna(0)
    df["has_size"] = pd.to_numeric(df.get("has_size", 0), errors="coerce").fillna(0).astype(int)

    # Categorical / text
    cat_cols = ["style", "skin_temperature", "skin_depth", "frame", "height_bucket", "shoulders"]
    text_cols = ["color_tags", "fit_tags", "avoid_tags"]
    for c in text_cols:
        df[c] = df.get(c, "").fillna("")

    X = df[["price", "has_size"] + cat_cols + text_cols]
    y = df["label"].astype(int)

    # Preprocessors
    ohe = OneHotEncoder(handle_unknown="ignore")

    bow_color = Pipeline([
        ("prep", FunctionTransformer(prep_text_col, validate=False)),
        ("vec", CountVectorizer(tokenizer=split_semicolon, token_pattern=None, binary=True)),
    ])
    bow_fit = Pipeline([
        ("prep", FunctionTransformer(prep_text_col, validate=False)),
        ("vec", CountVectorizer(tokenizer=split_semicolon, token_pattern=None, binary=True)),
    ])
    bow_avoid = Pipeline([
        ("prep", FunctionTransformer(prep_text_col, validate=False)),
        ("vec", CountVectorizer(tokenizer=split_semicolon, token_pattern=None, binary=True)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", ["price", "has_size"]),
            ("cat", ohe, cat_cols),
            ("bow_color", bow_color, ["color_tags"]),
            ("bow_fit", bow_fit, ["fit_tags"]),
            ("bow_avoid", bow_avoid, ["avoid_tags"]),
        ],
        remainder="drop",
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"))
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(X_tr, y_tr)

    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    print(f"[train_model] Validation AUC: {auc:.3f}")

    joblib.dump(pipe, MODEL_PATH)
    print(f"[train_model] Saved model → {MODEL_PATH}")

if __name__ == "__main__":
    main()
