# # import requests
# # from bs4 import BeautifulSoup
# # import json
# # import time

# # BASE_URL = "https://www.shl.com"
# # # This is the main catalog path; your snippet suggests there's a `start` param and maybe `type=2`.
# # CATALOG_PATH = "/solutions/products/product-catalog/"

# # TOTAL_PAGES = 12           # Number of pages you want to scrape
# # ITEMS_PER_PAGE = 20        # Adjust if each page shows 20 items, or a different number

# # def scrape_page(start_offset):
# #     """
# #     Fetches a single catalog page using the 'start' parameter for pagination.
# #     Extracts product rows and returns a list of product dicts.
# #     """
# #     page_url = f"{BASE_URL}{CATALOG_PATH}?start={start_offset}&type=2"
# #     print(f"Scraping: {page_url}")
    
# #     resp = requests.get(page_url)
# #     if resp.status_code != 200:
# #         print(f"[!] Could not fetch {page_url} (status {resp.status_code}). Skipping.")
# #         return []

# #     soup = BeautifulSoup(resp.text, "html.parser")

# #     # Rows may have data-entity-id or data-course-id; check your actual HTML
# #     rows = soup.select("tr[data-entity-id]") or soup.select("tr[data-course-id]")
# #     print(f"Found {len(rows)} product rows at offset={start_offset}.")

# #     products = []
# #     for row in rows:
# #         # Example: product name in a <td> with class="custom__table-heading__title"
# #         name_td = row.find("td", class_="custom__table-heading__title")
# #         if not name_td:
# #             continue

# #         product_name = name_td.get_text(strip=True)

# #         # If there's an <a> tag for the product link
# #         a_tag = name_td.find("a")
# #         product_url = None
# #         if a_tag:
# #             href = a_tag.get("href", "").strip()
# #             if href.startswith("/"):
# #                 product_url = BASE_URL + href
# #             else:
# #                 product_url = href

# #         products.append({
# #             "Name": product_name,
# #             "URL": product_url
# #         })

# #     return products

# # def main():
# #     all_products = []
# #     # For each page, compute the 'start' offset
# #     for page_num in range(1, TOTAL_PAGES + 1):
# #         start_offset = (page_num - 1) * ITEMS_PER_PAGE
# #         page_products = scrape_page(start_offset)
# #         all_products.extend(page_products)
# #         time.sleep(1)  # Polite delay

# #     # Write to a JSONL file
# #     output_file = "shl_products_correct_pagination.jsonl"
# #     with open(output_file, "w", encoding="utf-8") as f:
# #         for product in all_products:
# #             f.write(json.dumps(product, ensure_ascii=False) + "\n")

# #     print(f"\nDone! Scraped {len(all_products)} products from {TOTAL_PAGES} pages.")
# #     print(f"Data saved to '{output_file}'.")

# # if __name__ == "__main__":
# #     main()
# import requests
# from bs4 import BeautifulSoup
# import json
# import time

# BASE_URL = "https://www.shl.com"
# CATALOG_PATH = "/solutions/products/product-catalog/"

# # We assume there are 12 pages total and each page increments start by 12
# TOTAL_PAGES = 32
# ITEMS_PER_PAGE = 12

# def scrape_page(start_offset):
#     """
#     Fetches a single catalog page using the 'start' parameter for pagination.
#     Extracts product rows and returns a list of product dicts (Name, URL).
#     """
#     # The screenshot suggests something like ?start=12&type=2 for page 2
#     page_url = f"{BASE_URL}{CATALOG_PATH}?start={start_offset}&type=1"
#     print(f"Scraping: {page_url}")

#     resp = requests.get(page_url)
#     if resp.status_code != 200:
#         print(f"[!] Could not fetch {page_url} (status {resp.status_code}). Skipping.")
#         return []

#     soup = BeautifulSoup(resp.text, "html.parser")

#     # Rows may have data-entity-id or data-course-id; check your HTML to confirm
#     rows = soup.select("tr[data-entity-id]") or soup.select("tr[data-course-id]")
#     print(f"Found {len(rows)} product rows at start={start_offset}.")

#     products = []
#     for row in rows:
#         # Example: product name in <td class="custom__table-heading__title">
#         name_td = row.find("td", class_="custom__table-heading__title")
#         if not name_td:
#             continue

#         product_name = name_td.get_text(strip=True)

#         # If there's an <a> tag for the product link
#         a_tag = name_td.find("a")
#         product_url = None
#         if a_tag:
#             href = a_tag.get("href", "").strip()
#             if href.startswith("/"):
#                 product_url = BASE_URL + href
#             else:
#                 product_url = href

#         products.append({
#             "Name": product_name,
#             "URL": product_url
#         })

#     return products

# def main():
#     all_products = []
#     # page_num goes from 0..11; offset = page_num*12 => 0,12,24,... for 12 pages
#     for page_num in range(TOTAL_PAGES):
#         start_offset = page_num * ITEMS_PER_PAGE
#         page_products = scrape_page(start_offset)
#         all_products.extend(page_products)
#         time.sleep(1)  # Polite delay

#     # Write to a JSONL file
#     output_file = "shl_products_12pages.jsonl"
#     with open(output_file, "w", encoding="utf-8") as f:
#         for product in all_products:
#             f.write(json.dumps(product, ensure_ascii=False) + "\n")

#     print(f"\nDone! Scraped {len(all_products)} products from {TOTAL_PAGES} pages.")
#     print(f"Data saved to '{output_file}'.")

# if __name__ == "__main__":
#     main()

# import requests
# from bs4 import BeautifulSoup
# import json
# import time

# BASE_URL = "https://www.shl.com"
# # This is the main catalog path; your snippet suggests there's a `start` param and maybe `type=2`.
# CATALOG_PATH = "/solutions/products/product-catalog/"

# TOTAL_PAGES = 12           # Number of pages you want to scrape
# ITEMS_PER_PAGE = 20        # Adjust if each page shows 20 items, or a different number

# def scrape_page(start_offset):
#     """
#     Fetches a single catalog page using the 'start' parameter for pagination.
#     Extracts product rows and returns a list of product dicts.
#     """
#     page_url = f"{BASE_URL}{CATALOG_PATH}?start={start_offset}&type=2"
#     print(f"Scraping: {page_url}")
    
#     resp = requests.get(page_url)
#     if resp.status_code != 200:
#         print(f"[!] Could not fetch {page_url} (status {resp.status_code}). Skipping.")
#         return []

#     soup = BeautifulSoup(resp.text, "html.parser")

#     # Rows may have data-entity-id or data-course-id; check your actual HTML
#     rows = soup.select("tr[data-entity-id]") or soup.select("tr[data-course-id]")
#     print(f"Found {len(rows)} product rows at offset={start_offset}.")

#     products = []
#     for row in rows:
#         # Example: product name in a <td> with class="custom__table-heading__title"
#         name_td = row.find("td", class_="custom__table-heading__title")
#         if not name_td:
#             continue

#         product_name = name_td.get_text(strip=True)

#         # If there's an <a> tag for the product link
#         a_tag = name_td.find("a")
#         product_url = None
#         if a_tag:
#             href = a_tag.get("href", "").strip()
#             if href.startswith("/"):
#                 product_url = BASE_URL + href
#             else:
#                 product_url = href

#         products.append({
#             "Name": product_name,
#             "URL": product_url
#         })

#     return products

# def main():
#     all_products = []
#     # For each page, compute the 'start' offset
#     for page_num in range(1, TOTAL_PAGES + 1):
#         start_offset = (page_num - 1) * ITEMS_PER_PAGE
#         page_products = scrape_page(start_offset)
#         all_products.extend(page_products)
#         time.sleep(1)  # Polite delay

#     # Write to a JSONL file
#     output_file = "shl_products_correct_pagination.jsonl"
#     with open(output_file, "w", encoding="utf-8") as f:
#         for product in all_products:
#             f.write(json.dumps(product, ensure_ascii=False) + "\n")

#     print(f"\nDone! Scraped {len(all_products)} products from {TOTAL_PAGES} pages.")
#     print(f"Data saved to '{output_file}'.")

# if __name__ == "__main__":
#     main()
import requests
from bs4 import BeautifulSoup
import json
import time

BASE_URL = "https://www.shl.com"
CATALOG_PATH = "/solutions/products/product-catalog/"

# We assume there are 12 pages total and each page increments start by 12
TOTAL_PAGES = 12
ITEMS_PER_PAGE = 12

def scrape_page(start_offset):
    """
    Fetches a single catalog page using the 'start' parameter for pagination.
    Extracts product rows and returns a list of product dicts (Name, URL).
    """
    # The screenshot suggests something like ?start=12&type=2 for page 2
    page_url = f"{BASE_URL}{CATALOG_PATH}?start={start_offset}&type=2&type=2"
    print(f"Scraping: {page_url}")

    resp = requests.get(page_url)
    if resp.status_code != 200:
        print(f"[!] Could not fetch {page_url} (status {resp.status_code}). Skipping.")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Rows may have data-entity-id or data-course-id; check your HTML to confirm
    rows = soup.select("tr[data-entity-id]") or soup.select("tr[data-course-id]")
    print(f"Found {len(rows)} product rows at start={start_offset}.")

    products = []
    for row in rows:
        # Example: product name in <td class="custom__table-heading__title">
        name_td = row.find("td", class_="custom__table-heading__title")
        if not name_td:
            continue

        product_name = name_td.get_text(strip=True)

        # If there's an <a> tag for the product link
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
    # page_num goes from 0..11; offset = page_num*12 => 0,12,24,... for 12 pages
    for page_num in range(TOTAL_PAGES):
        start_offset = page_num * ITEMS_PER_PAGE
        page_products = scrape_page(start_offset)
        all_products.extend(page_products)
        time.sleep(1)  # Polite delay

    # Write to a JSONL file
    output_file = "shl_products_2pages.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for product in all_products:
            f.write(json.dumps(product, ensure_ascii=False) + "\n")

    print(f"\nDone! Scraped {len(all_products)} products from {TOTAL_PAGES} pages.")
    print(f"Data saved to '{output_file}'.")

if __name__ == "__main__":
    main()
