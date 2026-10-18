"""Phase 2 - Exploratory data analysis. Saves charts to figures/ for the LaTeX report
and summary statistics to figures/eda_summary.txt."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CLEAN_PATH = "data/processed/products_clean.csv"
FEAT_PATH = "data/processed/products_features.csv"


def save(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=200)
    plt.close()


def countplot(df, col, name, title, top=None, size=(7, 4)):
    order = df[col].value_counts()
    order = order.head(top).index if top else order.index
    plt.figure(figsize=size)
    sns.countplot(data=df, y=col, order=order)
    plt.title(title)
    save(name)


def main():
    os.makedirs("figures", exist_ok=True)
    df = pd.read_csv(CLEAN_PATH, keep_default_na=False, na_values=[""])
    feats = pd.read_csv(FEAT_PATH)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4))
    sns.histplot(df["price"], bins=40, log_scale=True)
    plt.title("Price distribution (log scale)")
    plt.xlabel("Price (Rs.)")
    save("price_distribution")

    countplot(df, "category", "category_distribution", "Products per category")
    countplot(df, "color", "color_distribution", "Top 15 colors", top=15, size=(7, 5))
    countplot(df, "brand", "brand_distribution", "Products per brand")
    countplot(df, "gender", "gender_distribution", "Products per target gender")
    countplot(df, "formality", "formality_distribution", "Formality level")
    countplot(df, "fit_type", "fit_distribution", "Fit type")
    countplot(df, "color_undertone", "undertone_distribution", "Color undertone group")

    occ = df["occasions"].str.split("|").explode().rename("occasion").to_frame()
    plt.figure(figsize=(7, 4))
    sns.countplot(data=occ, y="occasion", order=occ["occasion"].value_counts().index)
    plt.title("Occasion labels (multi-label)")
    save("occasion_distribution")

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="category", y="price")
    plt.yscale("log")
    plt.title("Price by category (log scale)")
    save("price_by_category")

    plus = df.groupby("category")["has_plus_sizes"].mean().mul(100).reset_index()
    plt.figure(figsize=(7, 4))
    sns.barplot(data=plus, x="category", y="has_plus_sizes")
    plt.ylabel("% with 2XL or larger")
    plt.title("Plus-size availability by category")
    save("plus_size_by_category")

    num_cols = ["price", "log_price", "discount_pct", "n_sizes", "in_stock", "is_stretch", "has_plus_sizes"]
    plt.figure(figsize=(7, 6))
    sns.heatmap(feats[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Feature correlations")
    save("feature_correlations")

    with open("figures/eda_summary.txt", "w", encoding="utf-8") as f:
        f.write(df[["price", "discount_pct"]].describe().to_string())
        f.write(f"\n\nRows: {len(df)}\nBrands: {df['brand'].nunique()}\n")
    print("Saved charts to figures/")


if __name__ == "__main__":
    main()
