# Trouvaille

AI-powered shopping web app that builds complete outfits from natural language queries
(e.g. "a black outfit under Rs. 15,000 for a concert"). An LLM extracts budget, color,
occasion and size; sentence embeddings match products from a cleaned, scraped dataset; an
inclusive re-ranker (skin tone, body type, ratings) and an outfit builder assemble a look
within budget; a Vibe Check Engine verifies it fits the event type.

## Deliverable 1
- **Phase 1:** problem definition (see `report/`)
- **Phase 2:** data collection, cleaning, feature preparation and EDA (`data_scripts/`)

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run (from the repo root)
```bash
python data_scripts/run_pipeline.py
```
or step by step:
```bash
python data_scripts/scrape_products.py       # -> data/raw/products_raw.csv, scrape_log.txt
python data_scripts/clean_data.py            # -> data/processed/products_clean.csv, cleaning_report.txt
python data_scripts/preprocess_features.py   # -> data/processed/products_features.csv
python data_scripts/eda.py                   # -> figures/*.png
```
Edit the `STORES` dictionary in `scrape_products.py` to choose brands. The scraper reads each
store's public Shopify `/products.json`, checks `robots.txt`, and rate-limits its requests.
Prices are assumed to be in PKR.

## Repository structure
```
data_scripts/   all Phase 2 code
report/         LaTeX source (main.tex, phase1.tex, phase2.tex)
figures/        EDA charts (generated)
data/           raw and processed datasets (generated, not committed)
models/         saved scalers/encoders (generated, not committed)
```

## Data dictionary (products_clean.csv)
| Column | Description |
|---|---|
| brand, title, url, image_url | Source information |
| category | set, top, bottom, outerwear, footwear, accessory (keyword-derived) |
| gender | men, women, unisex (kids removed) |
| color | Normalized main color |
| color_undertone, suits_skin_undertone | Warm/cool/neutral group and the skin undertones it is assumed to suit (color-analysis heuristic) |
| fit_type, is_stretch | fitted / relaxed / regular from keywords; stretch flag |
| formality | formal / smart_casual / casual (keyword heuristic) |
| occasions | Multi-label: wedding, office, presentation, concert, casual (keyword heuristic) |
| sizes, has_plus_sizes | Standardized sizes; 1 if 2XL or larger is offered |
| price, compare_at_price, discount_pct | Pricing |
| skin_tone_label_manual, body_type_label_manual | Empty placeholders for labels added later |

Derived labels are heuristics, not ground truth. Ratings are not collected yet.

## Team
Add group member names here.
