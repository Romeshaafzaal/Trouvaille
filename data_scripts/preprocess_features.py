"""Phase 2 - Feature strategy: scaling and encoding.
Input : data/processed/products_clean.csv
Output: data/processed/products_features.csv, models/preprocessors.joblib

Scaling : price -> MinMax [0,1]; log(price) -> standardized (z-score); discount -> MinMax.
Encoding: category, gender, grouped color, brand, color undertone, fit type, formality -> one-hot;
          occasions and sizes (multi-label) -> multi-hot.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, StandardScaler

IN_PATH = "data/processed/products_clean.csv"
OUT_PATH = "data/processed/products_features.csv"
TOP_COLORS = 12  # rarer colors are grouped as "other_color"


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(IN_PATH, keep_default_na=False, na_values=[""])
    df["compare_at_price"] = pd.to_numeric(df["compare_at_price"], errors="coerce")

    # Numeric features and scaling
    df["log_price"] = np.log1p(df["price"])
    df["n_sizes"] = df["sizes"].apply(lambda s: 0 if s in ("UNKNOWN", "ONE SIZE") else len(str(s).split("|")))
    df["in_stock"] = df["available"].astype(int)
    minmax_price = MinMaxScaler()
    df["price_minmax"] = minmax_price.fit_transform(df[["price"]]).ravel()
    zscore = StandardScaler()
    df["log_price_z"] = zscore.fit_transform(df[["log_price"]]).ravel()
    minmax_disc = MinMaxScaler()
    df["discount_minmax"] = minmax_disc.fit_transform(df[["discount_pct"]]).ravel()

    # One-hot encoding of single-label categoricals
    top = df["color"].value_counts().head(TOP_COLORS).index
    df["color_grouped"] = df["color"].where(df["color"].isin(top), "other_color")
    cat_cols = ["category", "gender", "color_grouped", "brand", "color_undertone", "fit_type", "formality"]
    prefixes = ["cat", "gender", "color", "brand", "undertone", "fit", "formality"]
    onehot = pd.get_dummies(df[cat_cols], prefix=prefixes, dtype=int)

    # Multi-hot encoding of multi-label columns
    occ_mlb = MultiLabelBinarizer()
    occ = pd.DataFrame(occ_mlb.fit_transform(df["occasions"].str.split("|")),
                       columns=[f"occ_{c}" for c in occ_mlb.classes_], index=df.index)
    size_mlb = MultiLabelBinarizer()
    sizes = pd.DataFrame(size_mlb.fit_transform(df["sizes"].str.split("|")),
                         columns=[f"size_{c}" for c in size_mlb.classes_], index=df.index)

    out = pd.concat([df, onehot, occ, sizes], axis=1)
    out.to_csv(OUT_PATH, index=False)
    joblib.dump({"minmax_price": minmax_price, "zscore_log_price": zscore, "minmax_discount": minmax_disc,
                 "occasion_binarizer": occ_mlb, "size_binarizer": size_mlb, "top_colors": list(top)},
                "models/preprocessors.joblib")
    print(f"Saved {out.shape[0]} rows x {out.shape[1]} columns to {OUT_PATH}")


if __name__ == "__main__":
    main()
