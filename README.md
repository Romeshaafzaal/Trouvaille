# Trouvaille

AI-powered shopping web app that finds outfits from Pakistani stores by occasion, budget and personal filters.
A PyTorch classifier trained on sentence embeddings predicts which occasions each product suits. A free-text occasion
box is matched to the closest occasion with the same embeddings, and results are filtered by budget, color, size,
skin undertone and body type.

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
app.py          Streamlit app (Deliverable 3)
style.css       app styling
app_data/       products_app.csv: products with precomputed occasion scores (committed, used by the app)
assets/         logo and images
src/            Deliverable 2 code: split, baseline, MLP, evaluation, error analysis, prediction, app data prep
data_scripts/   all Phase 2 code (scrape, clean, features, EDA)
report/         LaTeX source and compiled PDFs
figures/        EDA and model charts (generated)
data/           raw and processed datasets (generated, not committed); only
                raw/scrape_log.txt and processed/cleaning_report.txt are kept as evidence
models/         saved scalers/encoders and model weights (generated, not committed)
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

## Model weights
Download: (https://drive.google.com/file/d/1EW09pk8n9WU_330j14oFjEJhDgPJFmbW/view?usp=sharing)
Place the file in `models/mlp_occasion.pt`.


## Deliverable 2: Model and Evaluation
Run from the repo root after the Deliverable 1 pipeline. Weights go in `models/mlp_occasion.pt` (see link above).

```bash
python src/split_data.py        # 70/15/15 stratified split
python src/baseline.py          # embeddings + logistic regression baseline
python src/mlp.py               # PyTorch MLP, grid search, saves models/mlp_occasion.pt
python src/evaluate.py          # test metrics + subgroup bias check
python src/error_analysis.py    # per-class metrics, ROC, confusion matrices
python src/plot_results.py      # tuning and bias figures/tables
python src/predict.py           # interactive demo
python src/contrib_table.py     # contribution table from git history
```

Results: macro F1 0.928 and AUC-ROC 0.980 on the test set (baseline validation F1 0.878). Report: `report/Deliverable2_Report.pdf`.

## Deliverable 3: Application (Track A)
Streamlit app in `app.py`, styled by `style.css`.

### Run the app
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
The app reads `app_data/products_app.csv`, which is committed and holds each product's occasion scores. The scores come
from the trained MLP via `python src/prepare_app_data.py`, so the app does not need the raw data or the weights file to run.

### Features
- Free-text occasion (matched to wedding, office, presentation, concert or casual with sentence embeddings)
- Budget range with input validation
- Filters: gender, type, color, size, skin undertone, body type, in stock
- Results ranked by the model's occasion score, with pagination

## Feedback 
Each result card has 👍 / 👎 buttons. Votes are stored locally in `data/feedback.db` (SQLite, not committed).
- Ranking: products get a small capped boost or penalty from feedback (`feedback.py`, max ±0.2).
- Insights: the "Community feedback" panel shows approval rate per occasion and the most and least liked products.

## Saved outfits
Each result card has a 🤍 Save button. Saved outfits appear in the sidebar with a link back to the store,
a remove button and a CSV download. The list is kept in the browser session only, so it resets on refresh.

### What is learned and what is rule-based
- Learned: the occasion classifier (PyTorch MLP) and the semantic matching of the typed occasion.
- Rule-based: skin undertone matching (color-analysis lookup), body type (fit keywords) and all other filters.

### Limitations
- The size filter shows products that offer a size, not whether that size is in stock.
- Skin undertone and body type are heuristic filters, not learned models.
- Occasion labels come from keyword rules.
- Not built yet: LLM query parsing, ratings re-ranking, outfit builder, Vibe Check Engine (future work).

#Trouvaille results page
<img width="924" height="446" alt="image" src="https://github.com/user-attachments/assets/3d037198-d623-461d-866f-18e399a1ccef" />


## Team
- Romesha Afzaal (24L-3001)
- Aliza Nadeem (24L-3024)
