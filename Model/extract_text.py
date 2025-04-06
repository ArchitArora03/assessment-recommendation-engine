import sys
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

def extract_all_text(url):
    """
    Use Selenium with headless Chrome to open the URL, wait for dynamic content to load,
    and extract all visible text from the page.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)
    
    driver.get(url)
    time.sleep(5)
    html = driver.page_source
    driver.quit()
    
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_text.py <URL>")
        sys.exit(1)
    url = sys.argv[1]
    extracted_text = extract_all_text(url)
    print("Extracted Text:")
    print(extracted_text)
