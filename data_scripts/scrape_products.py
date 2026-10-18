"""Phase 2 - Data collection.
Scrapes product listings from Shopify-based Pakistani clothing stores via the public
/products.json endpoint. It checks each store's robots.txt first and logs everything
(the log is your evidence of data gathering for the report).

Before running: confirm https://<store>/products.json loads in your browser and read the
store's terms of use. Edit STORES below to add/remove brands.

Output: data/raw/products_raw.csv, data/raw/scrape_log.txt
"""
import csv
import logging
import os
import time
from datetime import datetime, timezone
from urllib import robotparser

import requests

# brand -> base URL (examples only: verify each one, add/remove as needed)
STORES = {
    "Outfitters": "https://outfitters.com.pk",
    "Sapphire": "https://pk.sapphireonline.pk",
    "Bonanza Satrangi": "https://bonanzasatrangi.com",
}

RAW_PATH = "data/raw/products_raw.csv"
LOG_PATH = "data/raw/scrape_log.txt"
HEADERS = {"User-Agent": "Mozilla/5.0 (university AI project; contact: romeshaafzaal72@gmail.com)"}
DELAY_SECONDS = 1.5
MAX_PAGES = 30
RESPECT_ROBOTS = True

FIELDS = ["brand", "product_id", "title", "handle", "product_type", "vendor", "tags",
          "price", "compare_at_price", "sizes", "colors", "available", "image_url",
          "url", "description", "created_at", "scraped_at"]


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


def get_json(url, params, tries=3):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if r.status_code in (403, 404):
                logging.warning("%s returned %d (not available / blocked)", url, r.status_code)
                return None
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
    if not allowed_by_robots(base):
        logging.warning("Skipping %s because robots.txt disallows it", brand)
        return []
    rows = []
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for page in range(1, MAX_PAGES + 1):
        data = get_json(f"{base}/products.json", {"limit": 250, "page": page})
        products = (data or {}).get("products", [])
        if not products:
            break
        rows.extend(parse_product(brand, base, p, scraped_at) for p in products)
        logging.info("%s page %d: %d products (running total %d)", brand, page, len(products), len(rows))
        time.sleep(DELAY_SECONDS)
    logging.info("%s finished: %d products", brand, len(rows))
    return rows


def main():
    os.makedirs("data/raw", exist_ok=True)
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    all_rows = []
    for brand, base in STORES.items():
        all_rows.extend(fetch_store(brand, base))
    with open(RAW_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    logging.info("Saved %d raw products to %s", len(all_rows), RAW_PATH)
    if not all_rows:
        logging.error("No products collected. Check store URLs, /products.json access, or your internet.")


if __name__ == "__main__":
    main()
