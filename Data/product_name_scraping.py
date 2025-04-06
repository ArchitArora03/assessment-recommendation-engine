import requests
from bs4 import BeautifulSoup
import json
import time

BASE_URL = "https://www.shl.com"
CATALOG_PATH = "/solutions/products/product-catalog/"

TOTAL_PAGES = 12
ITEMS_PER_PAGE = 12

def scrape_page(start_offset):
    """
    Fetches a single catalog page using the 'start' parameter for pagination.
    Extracts product rows and returns a list of product dicts (Name, URL).
    """
    page_url = f"{BASE_URL}{CATALOG_PATH}?start={start_offset}&type=2&type=2"
    print(f"Scraping: {page_url}")

    resp = requests.get(page_url)
    if resp.status_code != 200:
        print(f"[!] Could not fetch {page_url} (status {resp.status_code}). Skipping.")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    rows = soup.select("tr[data-entity-id]") or soup.select("tr[data-course-id]")
    print(f"Found {len(rows)} product rows at start={start_offset}.")

    products = []
    for row in rows:
        name_td = row.find("td", class_="custom__table-heading__title")
        if not name_td:
            continue

        product_name = name_td.get_text(strip=True)

        a_tag = name_td.find("a")
        product_url = None
        if a_tag:
            href = a_tag.get("href", "").strip()
            if href.startswith("/"):
                product_url = BASE_URL + href
            else:
                product_url = href

        products.append({
            "Name": product_name,
            "URL": product_url
        })

    return products

def main():
    all_products = []
    for page_num in range(TOTAL_PAGES):
        start_offset = page_num * ITEMS_PER_PAGE
        page_products = scrape_page(start_offset)
        all_products.extend(page_products)
        time.sleep(1)  # Polite delay

    output_file = "shl_products_2pages.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for product in all_products:
            f.write(json.dumps(product, ensure_ascii=False) + "\n")

    print(f"\nDone! Scraped {len(all_products)} products from {TOTAL_PAGES} pages.")
    print(f"Data saved to '{output_file}'.")

if __name__ == "__main__":
    main()
