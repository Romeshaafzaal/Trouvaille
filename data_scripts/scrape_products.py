"""Phase 2 - Data collection.
Scrapes product listings from Shopify-based Pakistani clothing stores via the public
/products.json endpoint. It checks each store's robots.txt first and logs everything
(the log is your evidence of data gathering for the report).

Features: per-store caching (a finished store is not downloaded again, so re-runs do not
hammer the sites), patient handling of "429 Too Many Requests", and it never overwrites
your raw file with an empty result.

Usage:
    python data_scripts/scrape_products.py             # uses cached stores, scrapes the rest
    python data_scripts/scrape_products.py --refresh   # re-download every store

Output: data/raw/products_raw.csv, data/raw/stores/<brand>.csv, data/raw/scrape_log.txt
"""
import argparse
import csv
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from urllib import robotparser

import requests
 
STORES = {
    "Outfitters": "https://outfitters.com.pk",
    "Ethnc": "https://pk.ethnc.com",
    "Bonanza Satrangi": "https://bonanzasatrangi.com",
    "Lama": "https://lamaretail.com",
    "Lulusar": "https://lulusar.com",
    "Faiza Saqlain": "https://www.faizasaqlain.pk",
    "Hussain Rehar": "https://hussainrehar.com",
    "Saira Shakira": "https://www.sairashakira.com",
    "J.": "https://www.junaidjamshed.com",
    "Vanya": "https://vanya.pk",
}

RAW_PATH = "data/raw/products_raw.csv"
CACHE_DIR = "data/raw/stores"
LOG_PATH = "data/raw/scrape_log.txt"
HEADERS = {"User-Agent": "Mozilla/5.0 (university AI project; contact: your_email@example.com)"}
DELAY_SECONDS = 2.5
MAX_PAGES = 50         # 250 products per page
RESPECT_ROBOTS = True

FIELDS = ["brand", "product_id", "title", "handle", "product_type", "vendor", "tags",
          "price", "compare_at_price", "sizes", "colors", "available", "image_url",
          "url", "description", "created_at", "scraped_at"]


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def allowed_by_robots(base):
    if not RESPECT_ROBOTS:
        return True
    rp = robotparser.RobotFileParser()
    rp.set_url(f"{base}/robots.txt")
    try:
        rp.read()
    except Exception as e:
        logging.warning("Could not read robots.txt for %s (%s); proceeding carefully", base, e)
        return True
    ok = rp.can_fetch(HEADERS["User-Agent"], f"{base}/products.json")
    logging.info("robots.txt check for %s: %s", base, "allowed" if ok else "DISALLOWED")
    return ok


def get_json(url, params, tries=4):
    """Returns parsed JSON, or None if the request failed for good."""
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code in (403, 404):
                logging.warning("%s returned %d (not available / blocked)", url, r.status_code)
                return None
            if r.status_code == 429:  # rate limited: wait politely, honor Retry-After
                try:
                    wait = int(r.headers.get("Retry-After", 0))
                except ValueError:
                    wait = 0
                wait = min(max(wait, 30 * (2 ** attempt)), 180)
                logging.warning("429 Too Many Requests; waiting %ds (attempt %d/%d)", wait, attempt + 1, tries)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2 ** attempt
            logging.warning("Request failed (%s); retry %d/%d in %ds", e, attempt + 1, tries, wait)
            time.sleep(wait)
    return None


def parse_product(brand, base, p, scraped_at):
    variants = p.get("variants", [])
    prices = [float(v["price"]) for v in variants if v.get("price")]
    compare = [float(v["compare_at_price"]) for v in variants if v.get("compare_at_price")]
    options = {o["name"].lower(): o.get("values", []) for o in p.get("options", [])}
    tags = p.get("tags", "")
    if isinstance(tags, list):
        tags = "|".join(tags)
    images = p.get("images", [])
    return {
        "brand": brand,
        "product_id": p.get("id"),
        "title": p.get("title"),
        "handle": p.get("handle"),
        "product_type": p.get("product_type"),
        "vendor": p.get("vendor"),
        "tags": tags,
        "price": min(prices) if prices else None,
        "compare_at_price": min(compare) if compare else None,
        "sizes": "|".join(options.get("size", [])),
        "colors": "|".join(options.get("color", []) or options.get("colour", [])),
        "available": any(v.get("available") for v in variants),
        "image_url": images[0]["src"] if images else None,
        "url": f"{base}/products/{p.get('handle')}",
        "description": p.get("body_html"),
        "created_at": p.get("created_at"),
        "scraped_at": scraped_at,
    }


def fetch_store(brand, base):
    """Returns (rows, complete). complete=False if a request failed part-way."""
    if not allowed_by_robots(base):
        logging.warning("Skipping %s because robots.txt disallows it", brand)
        return [], False
    rows = []
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    complete = True
    for page in range(1, MAX_PAGES + 1):
        data = get_json(f"{base}/products.json", {"limit": 250, "page": page})
        if data is None:
            complete = False
            logging.error("%s: stopped at page %d after repeated failures", brand, page)
            break
        products = data.get("products", [])
        if not products:
            break
        rows.extend(parse_product(brand, base, p, scraped_at) for p in products)
        logging.info("%s page %d: %d products (running total %d)", brand, page, len(products), len(rows))
        time.sleep(DELAY_SECONDS)
    else:
        logging.warning("%s: hit MAX_PAGES (%d); catalog may be larger. Raise MAX_PAGES.", brand, MAX_PAGES)
    logging.info("%s finished: %d products%s", brand, len(rows), "" if complete else " (INCOMPLETE)")
    return rows, complete


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download all stores (ignore cache)")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())

    all_rows = []
    for brand, base in STORES.items():
        base = base.strip().rstrip("/")
        cache = os.path.join(CACHE_DIR, f"{slug(brand)}.csv")
        if not args.refresh and os.path.exists(cache):
            cached = read_csv(cache)
            if cached:
                logging.info("%s: using cached copy (%d products). Use --refresh to re-download.", brand, len(cached))
                all_rows.extend(cached)
                continue
        rows, complete = fetch_store(brand, base)
        if rows and complete:
            write_csv(cache, rows)
        all_rows.extend(rows)

    if not all_rows:
        logging.error("No products collected, so the raw file was NOT overwritten. "
                      "Likely rate-limited (429) or wrong URLs. Wait 30-60 minutes and re-run.")
        sys.exit(1)

    write_csv(RAW_PATH, all_rows)
    logging.info("Saved %d raw products to %s", len(all_rows), RAW_PATH)
    brands = {}
    for r in all_rows:
        brands[r["brand"]] = brands.get(r["brand"], 0) + 1
    logging.info("Products per brand: %s", brands)


if __name__ == "__main__":
    main()