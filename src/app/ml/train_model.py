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
import joblib

# import picklable helpers from a real module (so joblib can load them later)
from .preprocess_utils import prep_text_col, split_semicolon

ML_DIR = Path(__file__).resolve().parent
INPUT_CSV = ML_DIR / "events_training_data.csv"
MODEL_PATH = ML_DIR / "reco_lr.joblib"


def main():
    # 1) Expect a dataset
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Training CSV not found at {INPUT_CSV}. "
            "Please generate it first (e.g., run synthetic_data_generator.py) or provide your own dataset."
        )

    df = pd.read_csv(INPUT_CSV)

    if "label" not in df.columns:
        raise ValueError("CSV must include a 'label' column with 1/0.")

    # 2) Clean/normalize raw columns
    df["price"] = pd.to_numeric(df.get("price", 0), errors="coerce").fillna(0)
    df["has_size"] = pd.to_numeric(df.get("has_size", 0), errors="coerce").fillna(0).astype(int)

    cat_cols = ["style", "skin_temperature", "skin_depth", "frame", "height_bucket", "shoulders"]
    text_cols = ["color_tags", "fit_tags", "avoid_tags"]
    for c in text_cols:
        df[c] = df.get(c, "").fillna("")

    # Features (X) and target (y)
    X = df[["price", "has_size"] + cat_cols + text_cols]
    y = df["label"].astype(int)

    # 3) Build a preprocessing recipe per column type
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

    # 4) Wrap preprocessing + model into a single Pipeline
    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
    ])

    # 5) Train/validate
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(X_tr, y_tr)

    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    print(f"[train_model] Validation AUC: {auc:.3f}")

    # 6) Save the whole thing
    joblib.dump(pipe, MODEL_PATH)
    print(f"[train_model] Saved model → {MODEL_PATH}")


if __name__ == "__main__":
    main()
