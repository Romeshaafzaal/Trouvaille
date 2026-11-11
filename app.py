import pandas as pd
import streamlit as st

st.set_page_config(page_title="Trouvaille", page_icon="👗", layout="wide") 

@st.cache_data
def load():
    return pd.read_csv("app_data/products_app.csv")

df = load()
occasions = ["wedding", "office", "presentation", "concert", "casual"]

st.title("Trouvaille")
st.caption("Tell us the occasion and your budget. We'll find matching outfits from Pakistani stores.")

def parse_price(text):
    """Return an int, or None if the text isn't a valid price."""
    text = text.replace(",", "").replace("Rs.", "").replace("Rs", "").strip()
    return int(text) if text.isdigit() else None

# Inputs
c1, c2, c3, c4, c5 = st.columns(5)
occasion = c1.selectbox("Occasion", occasions)
min_text = c2.text_input("Min budget (Rs.)", value="0")
max_text = c3.text_input("Max budget (Rs.)", value="15000")
gender = c4.selectbox("For", ["any", "women", "men"])
category = c5.selectbox("Type", ["any"] + sorted(df["category"].dropna().unique()))
in_stock = st.checkbox("In stock only", value=True)
colors = sorted(c for c in df["color"].dropna().unique() if c != "unknown")
color = st.selectbox("Outfit color", ["any"] + colors)
size_order = ["XXS", "XS", "S", "M", "L", "XL", "2XL", "3XL", "4XL"]
available_sizes = {s for row in df["sizes"].dropna() for s in row.split("|")}
size_options = [s for s in size_order if s in available_sizes]
size = st.selectbox("Size", ["any"] + size_options)

# Validation
minBudget = parse_price(min_text)
maxBudget = parse_price(max_text)
if minBudget is None or maxBudget is None:
    st.error("Please enter prices as whole numbers, e.g. 5000.")
    st.stop()
if minBudget > maxBudget:
    st.error("Min budget can't be higher than max budget.")
    st.stop()
if maxBudget > 500000:
    st.warning("That budget is unusually high. Please check it.")
    st.stop()

# Filter and rank
res = df[df["price"].between(minBudget, maxBudget)]
if gender != "any":
    res = res[res["gender"] == gender]
if category != "any":
    res = res[res["category"] == category]
if color != "any":
    res = res[res["color"] == color]
if in_stock:
    res = res[res["available"] == True]
res = res[res[f"p_{occasion}"] >= 0.5].sort_values(f"p_{occasion}", ascending=False)
if size != "any":
    res = res[res["sizes"].fillna("").apply(lambda s: size in s.split("|"))]

st.subheader(f"{len(res)} matches for {occasion} under Rs. {maxBudget:,} and above {minBudget}")

# Show top 30 as cards (3 per row)
top = res.head(30).reset_index(drop=True)
for i in range(0, len(top), 3):
    cols = st.columns(3)
    for col, (_, r) in zip(cols, top.iloc[i:i + 3].iterrows()):
        with col:
            if isinstance(r["image_url"], str):
                st.image(r["image_url"], use_container_width=True)
            st.markdown(f"**{r['title']}**")
            st.write(f"{r['brand']} · {r['category']} · {r['color']}")
            price = f"Rs. {r['price']:,.0f}"
            if pd.notna(r["discount_pct"]) and r["discount_pct"] > 0:
                price += f"  (-{r['discount_pct']:.0f}%)"
            st.write(price)
            st.progress(float(r[f"p_{occasion}"]), text=f"{occasion} match: {r[f'p_{occasion}']:.0%}")
            st.link_button("View in store", r["url"])