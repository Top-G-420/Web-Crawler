import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from newspaper import Article
import nltk
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse

# -------------------------- Supabase Client --------------------------
from supabase import create_client, Client

# YOUR CREDENTIALS (keep them secret in real prod → use env vars!)
SUPABASE_URL = "https://aanenqfgmnuosyfbfxbk.supabase.co"
SUPABASE_KEY = "sb_secret_rZc47DbEIZpLf2nRrnj6ZA_FImuYy0C"  # This is your service_role key → full access

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------- Suppress warnings --------------------------
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# -------------------------- SCRAPER --------------------------
def main_scraper():
    site_urls = [
        'https://www.the-star.co.ke/',
        'https://www.tuko.co.ke/'
    ]
    max_articles = 6
    data = []

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => false});")

    for site in site_urls:
        print(f"\nLoading {site} ...")
        try:
            driver.get(site)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located(("tag name", "body")))
            time.sleep(8)
            html = driver.page_source
            links = extract_article_links(html, site, max_articles)
            print(f"   Found {len(links)} articles")

            for url in links:
                try:
                    print(f"   Fetching: {url}")
                    article = Article(url)
                    article.download()
                    time.sleep(2)
                    article.parse()

                    if article.publish_date and (datetime.now() - article.publish_date).days > 30:
                        continue

                    text = article.text.strip()
                    if len(text) < 150:
                        continue

                    data.append({
                        "site_url": site,
                        "article_url": url,
                        "title": article.title.strip() if article.title else "No title",
                        "publish_date": article.publish_date.date() if article.publish_date else None,
                        "keyword_category": categorize_article(text),
                        "full_text": text,
                        "summary_snippet": text[:200] + "..." if len(text) > 200 else text
                    })
                except Exception as e:
                    print(f"     Error fetching {url}: {e}")
                    continue
        except Exception as e:
            print(f"   Failed to load {site}: {e}")

    driver.quit()
    return pd.DataFrame(data) if data else None

# -------------------------- Helpers --------------------------
def categorize_article(text):
    if not text: return "Other"
    sentences = sent_tokenize(text.lower())
    scores = {"GBV": 0, "Cyberbullying": 0, "Scams": 0}
    for s in sentences:
        if any(k in s for k in ['gender based violence','gbv','domestic violence','femicide','sexual harassment']): scores["GBV"] += 1
        if any(k in s for k in ['cyberbullying','online harassment','trolling','social media abuse']): scores["Cyberbullying"] += 1
        if any(k in s for k in ['scam','fraud','phishing','fake investment']): scores["Scams"] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"

def extract_article_links(html, base_url, max_links=6):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        if urlparse(full).netloc == urlparse(base_url).netloc and any(x in full for x in ["/2025/", "/news/", "/article/", "/story/"]):
            links.add(full)
            if len(links) >= max_links:
                return list(links)
    return list(links)[:max_links]

# -------------------------- MAIN EXECUTION --------------------------
print("Step 1: Scraping articles...")
raw_df = main_scraper()

if raw_df is None or len(raw_df) == 0:
    print("No articles scraped. Exiting.")
else:
    print(f"Scraped {len(raw_df)} articles. Running analysis...")

    from transformers import pipeline

    ner = pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl", aggregation_strategy="simple")
    sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

    results = []
    for i, row in raw_df.iterrows():
        text = row["full_text"][:1400]
        print(f"   Analyzing {i+1}/{len(raw_df)}: {row['title'][:60]}...")

        # NER
        ents = ner(text)
        entities = " | ".join(set([f"{e['entity_group']}: {e['word']}" for e in ents if e['score'] > 0.8]))

        # Sentiment
        sent = sentiment(text[:512])[0]
        sentiment_label = sent['label'].capitalize()
        sentiment_score = round(sent['score'], 3)

        results.append({
            "entities": entities or "None",
            "sentiment": sentiment_label,
            "sentiment_score": sentiment_score
        })

    # Add analysis columns
    for col in results[0].keys():
        raw_df[col] = [r[col] for r in results]

    # Final DataFrame - exact match to Supabase table
    final_df = raw_df[[
        "site_url",
        "article_url",
        "title",
        "publish_date",
        "keyword_category",
        "summary_snippet",
        "full_text",
        "entities",
        "sentiment",
        "sentiment_score"
    ]].copy()

    # Clean publish_date for Supabase
    final_df['publish_date'] = pd.to_datetime(final_df['publish_date'], errors='coerce').dt.date

    # Save final CSV (optional backup)
    final_df.to_csv("analyzed_articles.csv", index=False, encoding="utf-8")
    print(f"Saved backup CSV with {len(final_df)} articles")

    # -------------------------- UPLOAD TO SUPABASE --------------------------
    print("Uploading to Supabase (upserting on article_url)...")
    records = final_df.to_dict(orient="records")

    try:
        response = supabase.table("articles") \
            .upsert(records, on_conflict="article_url") \
            .execute()

        inserted = len(response.data) if response.data else 0
        updated = len(records) - inserted
        print(f"SUCCESS → Inserted: {inserted}, Updated: {updated}")
    except Exception as e:
        print("Supabase upload failed:")
        print(e)

    print("\nAll done! Your Supabase `articles` table is now up to date.")
