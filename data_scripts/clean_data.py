"""Phase 2 - Cleaning and label derivation.
Handles duplicates, missing values and outliers, standardizes sizes/colors/categories, and
derives heuristic labels needed later by the pipeline (occasion, formality, fit, color
undertone, plus-size availability).

Input : data/raw/products_raw.csv
Output: data/processed/products_clean.csv, data/processed/cleaning_report.txt

NOTE: the derived labels are keyword/color heuristics, not ground truth. Document them as
such in the report. skin_tone_label_manual / body_type_label_manual are intentionally empty
placeholders for labels you add later (by hand or with a model).
"""
import os
import re

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/products_raw.csv"
OUT_PATH = "data/processed/products_clean.csv"
REPORT_PATH = "data/processed/cleaning_report.txt"

COLORS = ["off white", "black", "white", "grey", "gray", "navy", "blue", "red", "maroon",
          "burgundy", "green", "olive", "mint", "teal", "turquoise", "yellow", "mustard",
          "orange", "peach", "pink", "purple", "lilac", "brown", "tan", "khaki", "beige",
          "cream", "ivory", "gold", "silver"]

CATEGORY_KEYWORDS = {  # checked in this order, whole-word match
    "set": ["2 piece", "3 piece", "2-piece", "3-piece", "two piece", "three piece", "2pc", "3pc",
            "unstitched", "co-ord", "coord set", "suit", "shalwar kameez"],
    "footwear": ["shoe", "sneaker", "sandal", "slipper", "slide", "khussa", "boot", "loafer",
                 "heel", "flat", "kolhapuri", "pump"],
    "outerwear": ["jacket", "coat", "blazer", "shawl", "cardigan", "waistcoat", "parka", "puffer",
                  "bomber", "poncho"],
    "bottom": ["jean", "trouser", "pant", "shorts", "chino", "jogger", "skirt", "palazzo", "culotte",
               "shalwar", "denim", "cargo", "legging", "capri"],
    "top": ["shirt", "t-shirt", "tee", "polo", "kurta", "kurti", "top", "blouse", "hoodie",
            "sweatshirt", "sweater", "tunic", "dress", "kameez", "tank"],
    "accessory": ["bag", "belt", "watch", "wallet", "scarf", "cap", "hat", "sunglass", "jewel",
                  "earring", "necklace", "ring", "bracelet", "clutch", "dupatta", "stole", "sock"],
}

# Color-analysis style heuristic (approximation, document as such)
UNDERTONE_GROUPS = {
    "warm": ["red", "orange", "yellow", "mustard", "olive", "brown", "tan", "khaki", "peach", "gold",
             "cream", "ivory"],
    "cool": ["blue", "navy", "teal", "turquoise", "purple", "lilac", "mint", "silver", "grey",
             "burgundy"],
    "neutral": ["black", "white", "off white", "beige", "green", "pink", "maroon"],
}
UNDERTONE_SUITS = {"warm": "warm|neutral", "cool": "cool|neutral",
                   "neutral": "warm|cool|neutral", "unknown": "unknown"}

FIT_KEYWORDS = {  # prefix match, checked in order
    "fitted": ["slim", "skinny", "fitted", "bodycon", "tapered", "tailored", "cigarette"],
    "relaxed": ["relaxed", "oversized", "loose", "baggy", "wide", "boxy", "flared", "palazzo",
                "culotte"],
}
STRETCH_WORDS = ["stretch", "elastic", "spandex", "lycra"]

OCCASION_KEYWORDS = {  # prefix match on title + type + tags + start of description
    "wedding": ["wedding", "bridal", "festive", "embroider", "luxury", "silk", "velvet", "organza",
                "chiffon", "jamawar", "sherwani", "lehenga", "gharara", "zari", "party wear", "eid"],
    "office": ["formal", "blazer", "dress shirt", "oxford", "trouser", "chino", "waistcoat", "loafer",
               "office", "work", "button", "slacks", "cigarette pant"],
    "presentation": ["blazer", "dress shirt", "formal", "smart", "semi", "polo", "oxford", "loafer",
                     "trouser", "chino", "waistcoat", "button"],
    "concert": ["denim", "jean", "graphic", "tee", "t-shirt", "hoodie", "jacket", "sneaker", "boot",
                "leather", "cargo", "oversized", "jogger", "bomber", "printed", "street", "sweatshirt"],
    "casual": ["casual", "tee", "t-shirt", "polo", "shorts", "jogger", "hoodie", "sweatshirt", "denim",
               "jean", "sneaker", "sandal", "slipper", "cotton", "basic", "everyday"],
}
FORMAL_WORDS = ["formal", "blazer", "waistcoat", "sherwani", "embroider", "bridal", "wedding", "luxury",
                "silk", "velvet", "organza", "chiffon", "dress shirt", "oxford", "loafer", "tuxedo"]
SMART_WORDS = ["chino", "trouser", "polo", "smart", "button", "semi", "blouse", "slacks"]

SIZE_MAP = {"EXTRA SMALL": "XS", "X-SMALL": "XS", "SMALL": "S", "MEDIUM": "M", "LARGE": "L",
            "EXTRA LARGE": "XL", "X-LARGE": "XL", "XXL": "2XL", "XXXL": "3XL", "XXXXL": "4XL",
            "DEFAULT TITLE": "ONE SIZE", "FREE SIZE": "ONE SIZE", "FREESIZE": "ONE SIZE",
            "STANDARD": "ONE SIZE", "OS": "ONE SIZE"}
PLUS_SIZES = {"2XL", "3XL", "4XL", "5XL"}


def word_match(text, kw):
    return re.search(rf"\b{re.escape(kw)}(?:e?s)?\b", text) is not None


def prefix_match(text, kw):
    return re.search(rf"\b{re.escape(kw)}", text) is not None


def strip_html(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text))).strip() if pd.notna(text) else ""


def extract_color(text):
    text = str(text).lower()
    for c in COLORS:
        if re.search(rf"\b{c}\b", text):
            return "grey" if c == "gray" else c
    return None


def assign_category(text):
    text = str(text).lower()
    for cat, words in CATEGORY_KEYWORDS.items():
        if any(word_match(text, w) for w in words):
            return cat
    return "other"


def infer_gender(text):
    if re.search(r"\b(kids?|boys?|girls?|infants?|toddlers?|juniors?|baby)\b", text):
        return "kids"
    if re.search(r"\b(women|woman|ladies|lady)\b", text):
        return "women"
    if re.search(r"\b(men|man|gents)\b", text):
        return "men"
    return "unisex"


def normalize_sizes(value):
    if pd.isna(value) or not str(value).strip():
        return "UNKNOWN"
    tokens = []
    for t in str(value).split("|"):
        t = re.sub(r"\s+", " ", t.strip().upper())
        if t:
            tokens.append(SIZE_MAP.get(t, t))
    return "|".join(dict.fromkeys(tokens)) or "UNKNOWN"


def undertone(color):
    for group, cols in UNDERTONE_GROUPS.items():
        if color in cols:
            return group
    return "unknown"


def fit_type(text, category):
    if category in ("footwear", "accessory"):
        return "not_applicable"
    for label, words in FIT_KEYWORDS.items():
        if any(prefix_match(text, w) for w in words):
            return label
    return "regular"


def formality(text):
    if any(prefix_match(text, w) for w in FORMAL_WORDS):
        return "formal"
    if any(prefix_match(text, w) for w in SMART_WORDS):
        return "smart_casual"
    return "casual"


def occasions(text):
    hits = [o for o, words in OCCASION_KEYWORDS.items() if any(prefix_match(text, w) for w in words)]
    return "|".join(hits) if hits else "casual"


def main():
    os.makedirs("data/processed", exist_ok=True)
    lines = []

    def say(msg=""):
        print(msg)
        lines.append(str(msg))

    df = pd.read_csv(RAW_PATH)
    say(f"Raw rows: {len(df)}")
    miss = df.isna().sum()
    say("Missing values per column (raw):")
    say(miss[miss > 0].to_string() if (miss > 0).any() else "none")

    # 1. Duplicates
    n = len(df)
    df = df.drop_duplicates(subset=["brand", "handle"])
    df = df.drop_duplicates(subset=["brand", "title", "price"])
    say(f"\nDuplicates removed: {n - len(df)}")

    # 2. Missing / invalid values
    n = len(df)
    df = df.dropna(subset=["title", "price"])
    df = df[df["price"] > 0].copy()
    say(f"Dropped (missing title/price or price <= 0): {n - len(df)}")

    df["title"] = df["title"].astype(str).str.strip()
    df["description"] = df["description"].apply(strip_html)
    df["product_type"] = df["product_type"].fillna("")
    df["tags"] = df["tags"].fillna("")
    meta = (df["title"] + " " + df["product_type"] + " " + df["tags"].str.replace("|", " ")).str.lower()
    text = meta + " " + df["description"].str.lower().str[:400]

    df["color"] = df["colors"].apply(extract_color)
    df["color"] = df["color"].fillna(df["title"].apply(extract_color)).fillna("unknown")
    n_unknown_color = int((df["color"] == "unknown").sum())
    say(f"Products with unknown color after filling from title: {n_unknown_color}")

    df["category"] = df["product_type"].apply(assign_category)
    mask = df["category"] == "other"
    df.loc[mask, "category"] = df.loc[mask, "title"].apply(assign_category)
    df["gender"] = meta.apply(infer_gender)
    df["sizes"] = df["sizes"].apply(normalize_sizes)
    df["available"] = df["available"].fillna(False).astype(bool)

    n = len(df)
    df = df[df["category"] != "other"]
    say(f"Dropped (no recognizable clothing category): {n - len(df)}")
    text = text.loc[df.index]
    n = len(df)
    kids = df["gender"] == "kids"
    df, text = df[~kids], text[~kids]
    say(f"Dropped (kids products, out of scope): {n - len(df)}")

    # 3. Derived labels (heuristics)
    df["color_undertone"] = df["color"].apply(undertone)
    df["suits_skin_undertone"] = df["color_undertone"].map(UNDERTONE_SUITS)
    df["fit_type"] = [fit_type(t, c) for t, c in zip(text, df["category"])]
    df["is_stretch"] = text.apply(lambda t: int(any(prefix_match(t, w) for w in STRETCH_WORDS)))
    df["formality"] = text.apply(formality)
    df["occasions"] = text.apply(occasions)
    df["has_plus_sizes"] = df["sizes"].apply(lambda s: int(any(t in PLUS_SIZES for t in s.split("|"))))
    df["skin_tone_label_manual"] = ""   # placeholder, fill later
    df["body_type_label_manual"] = ""   # placeholder, fill later

    df["discount_pct"] = np.where(df["compare_at_price"] > df["price"],
                                  (1 - df["price"] / df["compare_at_price"]) * 100, 0).round(1)

    # 4. Outliers: IQR on log(price) per category (global reference if category too small)
    df["log_price"] = np.log1p(df["price"])
    keep = pd.Series(True, index=df.index)
    for cat, grp in df.groupby("category"):
        ref = grp["log_price"] if len(grp) >= 20 else df["log_price"]
        q1, q3 = ref.quantile([0.25, 0.75])
        iqr = q3 - q1
        keep[grp.index] = grp["log_price"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    say(f"Removed price outliers (IQR rule on log price, per category): {int((~keep).sum())}")
    text = text[keep]
    df = df[keep].drop(columns=["log_price"])

    # 5. Text for sentence embeddings
    df["text_for_embedding"] = (df["title"] + ". " + df["category"] + ", " + df["color"] + ", " +
                                df["gender"] + ", " + df["fit_type"].str.replace("_", " ") + ", " +
                                df["formality"].str.replace("_", " ") + ", occasions: " +
                                df["occasions"].str.replace("|", ", ") + ". " + df["description"].str[:300])

    cols = ["brand", "product_id", "title", "category", "gender", "color", "color_undertone",
            "suits_skin_undertone", "fit_type", "is_stretch", "formality", "occasions", "sizes",
            "has_plus_sizes", "price", "compare_at_price", "discount_pct", "available", "image_url",
            "url", "skin_tone_label_manual", "body_type_label_manual", "text_for_embedding"]
    df[cols].to_csv(OUT_PATH, index=False)

    say(f"\nFinal clean rows: {len(df)}")
    for col in ["brand", "category", "gender", "formality", "fit_type", "color_undertone"]:
        say(f"\n{col} counts:\n{df[col].value_counts().to_string()}")
    say("\noccasion counts (multi-label):")
    say(df["occasions"].str.split("|").explode().value_counts().to_string())
    say(f"\nProducts with plus sizes (2XL+): {int(df['has_plus_sizes'].sum())}")
    say(f"Saved to {OUT_PATH}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
