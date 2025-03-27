import pandas as pd
import requests
import os
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setting up efficient logging
logging.basicConfig(
    filename="script.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

API_KEY = open('API_KEY').read()
SEARCH_ENGINE_ID = open('SEARCH_ENGINE_ID').read()
SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

file_path = "logos.csv"
df = pd.read_csv(file_path)

# Create directory for saving logos
os.makedirs("./logos", exist_ok=True)

# Skip already processed domains
processed_domains = set(file.replace("_", ".").replace(".png", "") for file in os.listdir("./logos"))

def setup_selenium():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    service = Service(executable_path="C:\\Program Files\\ChromeServices\\chromedriver-win64\\chromedriver.exe")  # Update with your chromedriver path
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# Helper functions
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def preprocess_domain(domain):
    domain = domain.strip()
    if not domain.startswith(("http://", "https://")):
        domain = f"https://{domain}"
    parsed_url = urlparse(domain)
    if not parsed_url.netloc:
        raise ValueError(f"Invalid domain: {domain}")
    return domain

def download_img(image_url, save_path):
    try:
        response = requests.get(image_url, stream=True, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
            logging.info(f"Downloaded image from {image_url}")
        else:
            logging.error(f"Error downloading image from {image_url}")
    except Exception as e:
        logging.error(f"Error downloading image from {image_url}: {e}")

def scrape_website_for_logo(url):
    if not is_valid_url(url):
        logging.warning(f"Invalid URL: {url}")
        return None

    try:
        logging.info(f"Attempting to scrape logo from {url} using BeautifulSoup...")
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for img in soup.find_all("img"):
            src = img.get("src", "")
            alt = img.get("alt", "").lower()
            if "logo" in alt or "logo" in src.lower():
                logging.info(f"Found logo using BeautifulSoup: {src}")
                return urljoin(url, src) if not src.startswith("http") else src

        logging.info(f"No static logo found on {url}. Switching to Selenium for dynamic scraping...")
        driver = setup_selenium()
        driver.get(url)
        time.sleep(2)  # Shortened sleep time for page load
        images = driver.find_elements(By.TAG_NAME, "img")
        for img in images:
            src = img.get_attribute("src")
            alt = img.get_attribute("alt").lower()
            if src and ("logo" in alt or "logo" in src.lower()):
                logging.info(f"Found logo using Selenium: {src}")
                driver.quit()
                return urljoin(url, src) if not src.startswith("http") else src

        driver.quit()
        logging.info(f"No logo found for {url} after both static and dynamic scraping.")
    except Exception as e:
        logging.error(f"Error while scraping {url}: {e}")
    return None

def search_first_link(domain):
    params = {
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": domain,
        "num": 1,
    }
    try:
        response = requests.get(SEARCH_URL, params=params, timeout=5)
        response.raise_for_status()
        results = response.json()
        if "items" in results:
            return results["items"][0]["link"]
    except Exception as e:
        logging.error(f"Error searching for the first link of {domain}: {e}")
    return None

def search_first_image(company_name):
    params = {
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": f"{company_name} logo",
        "searchType": "image",
        "num": 1,
    }
    try:
        response = requests.get(SEARCH_URL, params=params, timeout=5)
        response.raise_for_status()
        results = response.json()
        if "items" in results:
            return results["items"][0]["link"]
    except Exception as e:
        logging.error(f"Error searching for logo of {company_name}: {e}")
    return None

# Main function for processing each domain
def process_domain(domain):
    if domain in processed_domains:
        logging.info(f"Skipping {domain}, already processed.")
        return

    try:
        logging.info(f"Processing domain: {domain}")
        company_name = domain.split(".")[0]
        url = preprocess_domain(domain)
        logo_url = scrape_website_for_logo(url)
        if logo_url:
            filename = domain.replace(".", "_") + ".png"
            save_path = os.path.join("./logos", filename)
            download_img(logo_url, save_path)
            return

        first_link = search_first_link(domain)
        if first_link:
            logo_url = scrape_website_for_logo(first_link)
            if logo_url:
                filename = domain.replace(".", "_") + ".png"
                save_path = os.path.join("./logos", filename)
                download_img(logo_url, save_path)
                return

        first_image_url = search_first_image(company_name)
        if first_image_url:
            filename = domain.replace(".", "_") + ".png"
            save_path = os.path.join("./logos", filename)
            download_img(first_image_url, save_path)
        else:
            logging.warning(f"No image found for {domain}.")
    except Exception as e:
        logging.error(f"Error processing {domain}: {e}")

# Parallelization using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers based on system resources
    futures = {executor.submit(process_domain, domain): domain for domain in df["domain"]}
    for future in as_completed(futures):
        domain = futures[future]
        try:
            future.result()
        except Exception as e:
            logging.error(f"Error in thread for domain {domain}: {e}")
