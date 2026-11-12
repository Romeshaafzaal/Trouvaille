import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

def _flag_scroll():
    st.session_state["scroll_top"] = True

st.set_page_config(page_title="Trouvaille", page_icon="👗", layout="wide")  
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
# Lists used by the dropdowns
colors = sorted(c for c in df["color"].dropna().unique() if c != "unknown")
size_order = ["XXS", "XS", "S", "M", "L", "XL", "2XL", "3XL", "4XL"]
available_sizes = {s for row in df["sizes"].dropna() for s in row.split("|")}
size_options = [s for s in size_order if s in available_sizes]

# Row 1
c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
occasion = c1.selectbox("Occasion", occasions)
min_text = c2.text_input("Min budget (Rs.)", value="0")
max_text = c3.text_input("Max budget (Rs.)", value="15000")
gender = c4.selectbox("For", ["any", "women", "men"])
category = c5.selectbox("Type", ["any"] + sorted(df["category"].dropna().unique()))
color = c6.selectbox("Outfit color", ["any"] + colors)

# Row 2
r1, r2, r3, _, _, _ = st.columns(6, gap="small")
size = r1.selectbox("Size", ["any"] + size_options)
skin_tone = r2.selectbox("Skin undertone", ["any", "warm", "cool", "neutral"])
body_type = r3.selectbox("Body type", ["any", "Slim / fitted", "Regular", "Relaxed / loose", "Curvy / plus size"])

# Row 3
in_stock = st.checkbox("In stock only", value=True)

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
st.subheader(f"{len(res)} matches for {occasion} in your budget")
if skin_tone != "any":
    res = res[res["suits_skin_undertone"].fillna("").apply(lambda s: skin_tone in s.split("|"))]

clothing = ["set", "top", "bottom", "outerwear"]

if size != "any":
    has_size = res["sizes"].fillna("").apply(lambda s: size in s.split("|"))
    res = res[~res["category"].isin(clothing) | has_size]

if body_type != "any":
    if body_type == "Slim / fitted":
        ok = res["fit_type"] == "fitted"
    elif body_type == "Regular":
        ok = res["fit_type"] == "regular"
    elif body_type == "Relaxed / loose":
        ok = res["fit_type"] == "relaxed"
    else:  # Curvy / plus size
        ok = (res["has_plus_sizes"] == 1) | (res["is_stretch"] == 1)
    res = res[~res["category"].isin(clothing) | ok]

# Show top 30 as cards (3 per row)
per_page = 30
total_pages = max(1, -(-len(res) // per_page))  # round up
page = min(st.session_state.get("page", 1), total_pages)
st.session_state["page"] = page
start = (page - 1) * per_page
top = res.iloc[start:start + per_page].reset_index(drop=True)
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

# Page selector (bottom, narrow)
_, mid, _ = st.columns([5, 1.5, 5])
mid.number_input("Page", min_value=1, max_value=total_pages, step=1, key="page", on_change=_flag_scroll)
mid.caption(f"Page {page} of {total_pages}")

if st.session_state.pop("scroll_top", False):
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const main = doc.querySelector('[data-testid="stMain"]') || doc.querySelector('section.main');
        if (main) main.scrollTo({top: 0, behavior: 'smooth'});
        </script>
        """,
        height=0,
    )