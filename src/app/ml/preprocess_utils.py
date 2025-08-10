# src/app/ml/preprocess_utils.py
import numpy as np

def prep_text_col(X):
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    arr = np.ravel(X).astype(str)
    bad = (arr == "") | (arr == "nan") | (arr == "None") | (arr == "NONE")
    arr[bad] = "__none__"
    return arr

def split_semicolon(s: str):
    if s is None:
        return []
    s = str(s).strip().lower()
    if s == "":
        return ["__none__"]
    if s == "__none__":
        return ["__none__"]
    return [tok.strip() for tok in s.split(";") if tok.strip()]
