# Trouvaille

AI-powered shopping web app that builds complete outfits from natural language queries
(e.g. "a black outfit under Rs. 15,000 for a concert"). An LLM extracts budget, color,
occasion and size; sentence embeddings match products from a cleaned, scraped dataset; an
inclusive re-ranker (skin tone, body type, ratings) and an outfit builder assemble a look
within budget; a Vibe Check Engine verifies it fits the event type.

## Deliverable 1
- **Phase 1:** problem definition (see `report/`)
- **Phase 2:** data collection, cleaning, feature preparation and EDA (`data_scripts/`)
- **Report:** `report/Deliverable1_Report.pdf` (LaTeX source in `report/main.tex`)

## Requirements
- Python 3.10 or newer
- Internet connection (only needed for the scraping step)

## Setup
```bash
git clone https://github.com/Romeshaafzaal/Trouvaille.git
cd Trouvaille
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Before you run the scraper
The scraper identifies itself to each store with a contact email. Open
`data_scripts/scrape_products.py`, find the `HEADERS` line (near line 44) and replace the
placeholder address with your own real email:
```python
HEADERS = {"User-Agent": "Mozilla/5.0 (university AI project; contact: YOUR_EMAIL_HERE)"}
```
You can also edit the `STORES` dictionary in the same file to choose which brands to scrape.

## Run (from the repo root)
Run everything (scrape, clean, features, EDA):
```bash
python data_scripts/run_pipeline.py
```
Reuse an existing `data/raw/products_raw.csv` and skip scraping (only works after a first scrape):
```bash
python data_scripts/run_pipeline.py --skip-scrape
```
Or run the steps one by one:
```bash
python data_scripts/scrape_products.py       # -> data/raw/products_raw.csv, data/raw/scrape_log.txt
python data_scripts/clean_data.py            # -> data/processed/products_clean.csv, cleaning_report.txt
python data_scripts/preprocess_features.py   # -> data/processed/products_features.csv, models/preprocessors.joblib
python data_scripts/eda.py                   # -> figures/*.png, figures/eda_summary.txt
```
The raw and processed data are not stored in this repository, so a first run must scrape.
A full scrape takes about 10 minutes. The scraper reads each store's public Shopify
`/products.json`, checks `robots.txt`, waits between requests and retries failures. Each finished
store is cached in `data/raw/stores/`, so re-runs do not download it again. Use
`python data_scripts/scrape_products.py --refresh` to re-download everything.
Prices are assumed to be in PKR.

## Repository structure
```
data_scripts/   all Phase 2 code
report/         LaTeX source (main.tex, phase1.tex, phase2.tex) and the compiled PDF
figures/        EDA charts (generated)
data/           raw and processed datasets (generated, not committed); only
                raw/scrape_log.txt and processed/cleaning_report.txt are kept as evidence
models/         saved scalers/encoders (generated, not committed)
```

## Data summary
Scraped on 24 October 2026 from 10 Pakistani clothing brands: 40,537 raw products, reduced to
24,967 clean products after removing duplicates, invalid rows, uncategorized items, kids' products
and price outliers (see `data/processed/cleaning_report.txt`).

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

## Known limitations
- The J. catalog was capped at 50 pages (12,500 products) and may be larger.
- Faiza Saqlain and Saira Shakira keep only 4 and 3 products after cleaning, so the data
  effectively comes from eight brands, and the luxury/bridal segment is almost absent.
- Some Lama prices look unusually low and may use a different unit in the feed.
- Occasion, fit, formality and undertone labels are keyword heuristics, not ground truth.
- The data is a single snapshot taken during a sale period (66% of products carry a discount).

## Team
- Romesha Afzaal (24L-3001)
- Aliza Nadeem (24L-3024)
