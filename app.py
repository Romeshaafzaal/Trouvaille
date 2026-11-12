import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

def _flag_scroll():
    st.session_state["scroll_top"] = True

st.set_page_config(page_title="Trouvaille", page_icon="👗", layout="wide")  
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

import time

if "booted" not in st.session_state:
    splash = st.empty()
    splash.markdown("""
    <style>
    .splash {position:fixed; inset:0; z-index:99999; background:#FBF3F1;
             display:flex; flex-direction:column; align-items:center;
             justify-content:center; font-family:'VT323',monospace; color:#2D2223;}
    .splash h1 {font-size:4.5rem; color:#B76E79; margin:0;}
    .splash p {font-size:1.5rem; margin:6px 0 0 0;}
    .bar {width:260px; height:18px; border:2px solid #2D2223; margin-top:22px; padding:2px;}
    .bar div {height:100%; width:0; background:#B76E79; animation:fill 2.5s linear forwards;}
    @keyframes fill {to {width:100%;}}
    </style>
    <div class="splash">
        <h1>Trouvaille</h1>
        <p>loading your wardrobe...</p>
        <div class="bar"><div></div></div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(2.5)
    splash.empty()
    st.session_state["booted"] = True

@st.cache_data
def load():
    return pd.read_csv("app_data/products_app.csv")

df = load()
import numpy as np
from sentence_transformers import SentenceTransformer

OCCASION_DESC = {
    "wedding": "wedding, bridal, shaadi, mehndi, walima, formal festive event",
    "office": "office, work, professional job, business meeting",
    "presentation": "presentation, interview, graduation, convocation, conference, formal talk, ceremony",
    "concert": "concert, party, night out, music festival, club",
    "casual": "casual everyday outing, hangout, daily wear, picnic",
}

@st.cache_resource
def get_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

@st.cache_data
def occasion_vectors(descs):
    return get_encoder().encode(list(descs), normalize_embeddings=True)

def match_occasion(text):
    q = get_encoder().encode([text], normalize_embeddings=True)[0]
    sims = occasion_vectors(tuple(OCCASION_DESC.values())) @ q
    i = int(np.argmax(sims))
    return list(OCCASION_DESC)[i], float(sims[i])

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

# All filters in one row
c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9, gap="small")
occasion_text = c1.text_input("Occasion", value="casual")
min_text = c2.text_input("Min (Rs.)", value="0")
max_text = c3.text_input("Max (Rs.)", value="15000")
gender = c4.selectbox("For", ["any", "women", "men"])
category = c5.selectbox("Type", ["any"] + sorted(df["category"].dropna().unique()))
color = c6.selectbox(" Outfit Color", ["any"] + colors)
size = c7.selectbox("Size", ["any"] + size_options)
skin_tone = c8.selectbox("Undertone", ["any", "warm", "cool", "neutral"])
body_type = c9.selectbox("Body", ["any", "Slim / fitted", "Regular", "Relaxed / loose", "Curvy / plus size"])

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
occasion, score = match_occasion(occasion_text)
if score < 0.25:
    st.error("I couldn't understand that occasion. Try words like wedding, office, party or casual.")
    st.stop()

TONE_SCORES = {
    "warm":    {"mustard": 1, "rust": 1, "olive": 1, "coral": 1, "beige": .8, "navy": .4, "lavender": .3},
    "cool":    {"navy": 1, "emerald": 1, "lavender": 1, "maroon": .9, "white": .8, "mustard": .3, "rust": .3},
    "neutral": {},
}

def tone_score(color, tone):
    return TONE_SCORES.get(tone, {}).get(str(color).lower(), 0.6)

FIT_SCORES = {
    "Slim / fitted":     {"fitted": 1, "regular": .7, "relaxed": .4},
    "Regular":           {"regular": 1, "fitted": .7, "relaxed": .7},
    "Relaxed / loose":   {"relaxed": 1, "regular": .7, "fitted": .3},
    "Curvy / plus size": {"relaxed": .7, "regular": .6, "fitted": .4},
}

def fit_score(r, body):
    if body == "Curvy / plus size" and (r["has_plus_sizes"] == 1 or r["is_stretch"] == 1):
        return 1
    return FIT_SCORES.get(body, {}).get(r["fit_type"], 0.6)

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

res = res.copy()
res["rank"] = res[f"p_{occasion}"]

if skin_tone != "any":
    res["tone"] = res["color"].apply(lambda c: tone_score(c, skin_tone)).astype(float)
    res = res[res["tone"] >= 0.5]
    res["rank"] += 0.4 * res["tone"]

clothing = ["set", "top", "bottom", "outerwear"]

if size != "any":
    has_size = res["sizes"].fillna("").apply(lambda s: size in s.split("|"))
    res = res[~res["category"].isin(clothing) | has_size]

if body_type != "any":
    res = res.copy()
    res["fit"] = res.apply(lambda r: fit_score(r, body_type), axis=1).astype(float)
    res.loc[~res["category"].isin(clothing), "fit"] = 1  # non-clothing: neutral
    res = res[res["fit"] >= 0.5]
    res["rank"] += 0.4 * res["fit"]

res = res.sort_values("rank", ascending=False)

st.subheader(f"{len(res)} matches for {occasion_text} in your budget")

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

# Page selector (bottom): - and + buttons only
_, prev_col, mid, next_col, _ = st.columns([5, 1, 1.5, 1, 5])
if prev_col.button("−", key="prev_page", disabled=page <= 1):
    st.session_state["page"] = page - 1
    st.session_state["scroll_top"] = True
    st.rerun()
mid.markdown(f"<div style='text-align:center'>Page {page} of {total_pages}</div>", unsafe_allow_html=True)
if next_col.button("+", key="next_page", disabled=page >= total_pages):
    st.session_state["page"] = page + 1
    st.session_state["scroll_top"] = True
    st.rerun()
if st.session_state.pop("scroll_top", False):
    components.html(f"""
    <script>
    // unique every run: {page}-{time.time()}
    const doc = window.parent.document;
    function go() {{
        [doc.querySelector('[data-testid="stMain"]'),
         doc.querySelector('section.main'),
         doc.querySelector('[data-testid="stAppViewContainer"]'),
         doc.documentElement].forEach(t => {{ if (t) t.scrollTo({{top: 0}}); }});
    }}
    go(); setTimeout(go, 150); setTimeout(go, 400);
    </script>
    """, height=0)