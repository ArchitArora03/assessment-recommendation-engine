import requests
from bs4 import BeautifulSoup
import json
import time
import re

INPUT_FILE = "shl_products_12pages.jsonl"

OUTPUT_FILE = "final.jsonl"

def parse_detail_page(product):
    """
    Fetches the product detail page from its URL and extracts:
      - Description (from <h4>Description</h4> followed by a <p>)
      - Duration (from <h4>Assessment length</h4> followed by a <p>)
      - Job Levels (from <h4>Job levels</h4> followed by a <p>)
      - Test Type (from a <p> containing "Test Type:" with multiple <span class="product-catalogue__key"> elements)
      - Remote Testing is set to "Yes" by default.
    Updates the product dict in place.
    """
    url = product.get("URL", "")
    if not url:
        return product  

    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"[!] Could not fetch page: {url}")
            return product

        soup = BeautifulSoup(resp.text, "html.parser")


        desc_heading = soup.find("h4", string="Description")
        if desc_heading:
            desc_p = desc_heading.find_next_sibling("p")
            if desc_p:
                product["Description"] = desc_p.get_text(separator=" ", strip=True)

        
        duration_heading = soup.find("h4", string="Assessment length")
        if duration_heading:
            dur_p = duration_heading.find_next_sibling("p")
            if dur_p:
                text = dur_p.get_text(strip=True)
                match = re.search(r"(\d+)", text)
                if match:
                    product["Duration"] = match.group(1) + " minutes"
                else:
                    product["Duration"] = text

        
        job_levels_heading = soup.find("h4", string="Job levels")
        if job_levels_heading:
            jl_p = job_levels_heading.find_next_sibling("p")
            if jl_p:
                product["Job Levels"] = jl_p.get_text(separator=" ", strip=True)

    
        test_type_p = soup.find(lambda tag: tag.name == "p" and "Test Type:" in tag.get_text())
        if test_type_p:
            container = test_type_p.find("span", class_="d-flex ms-2")
            if container:
                key_spans = container.find_all("span", class_="product-catalogue__key")
                test_types = [ks.get_text(strip=True) for ks in key_spans]
                if test_types:
                    product["Test Type"] = ", ".join(test_types)
                else:
                    product["Test Type"] = None

        product["Remote Support"] = "Yes"

    except Exception as e:
        print(f"[!] Error parsing detail page for {url}: {e}")

    return product

def main():
    products = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                product = json.loads(line)
                products.append(product)
            except json.JSONDecodeError:
                print("[!] Error decoding a line in the input file.")

    print(f"[+] Loaded {len(products)} products from {INPUT_FILE}")

    
    enriched_products = []
    for i, product in enumerate(products, start=1):
        print(f"Enriching product {i}/{len(products)}: {product.get('Name', 'Unknown')}")
        updated_product = parse_detail_page(product)
        enriched_products.append(updated_product)
        time.sleep(1)  

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for prod in enriched_products:
            f.write(json.dumps(prod, ensure_ascii=False) + "\n")

    print(f"[+] Enrichment complete for {len(enriched_products)} products! Data saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()
