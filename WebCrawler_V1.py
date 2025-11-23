import os
import warnings
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Core imports
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

# NLP imports (with fallback if not installed)
try:
    from transformers import pipeline
    NER_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"
    SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    ner_pipeline = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple")
    sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    print("Transformers pipelines loaded successfully.")
except ImportError as e:
    print(f"Error: Install transformers with 'pip install transformers torch'. Details: {e}")
    ner_pipeline = None
    sentiment_pipeline = None
except Exception as e:
    print(f"Warning: Failed to load transformers pipelines: {e}")
    ner_pipeline = None
    sentiment_pipeline = None

# NLTK setup
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' data found.")
except LookupError:
    nltk.download('punkt', quiet=True)
    print("Downloaded NLTK 'punkt' data.")

# -------------------------- Supabase Client --------------------------
from supabase import create_client, Client

# For local PyCharm: Use env vars or hardcode temporarily
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://aanenqfgmnuosyfbfxbk.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_secret_rZc47DbEIZpLf2nRrnj6ZA_FImuYy0C")  # Service role key

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase client initialized successfully.")
except Exception as e:
    print(f"Supabase init failed: {e}. Check URL/key.")
    supabase = None

# -------------------------- HELPER FUNCTIONS --------------------------
def categorize_article(text):
    """Categorize article based on keywords in sentences."""
    if not text:
        return "Other"
    sentences = sent_tokenize(text.lower())
    scores = {"GBV": 0, "Cyberbullying": 0, "Scams": 0}
    gbv_keywords = ['gender based violence', 'gbv', 'domestic violence', 'femicide', 'sexual harassment']
    cyber_keywords = ['cyberbullying', 'online harassment', 'trolling', 'social media abuse']
    scam_keywords = ['scam', 'fraud', 'phishing', 'fake investment']
    for s in sentences:
        if any(k in s for k in gbv_keywords):
            scores["GBV"] += 1
        if any(k in s for k in cyber_keywords):
            scores["Cyberbullying"] += 1
        if any(k in s for k in scam_keywords):
            scores["Scams"] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"

def extract_article_links(html, base_url, max_links=6):
    """Extract article links from HTML soup, limited to same domain and specific paths."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    path_indicators = ["/news/", "/article/", "/story/"]  # Adjust if needed
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if (parsed.netloc == urlparse(base_url).netloc and
            any(indicator in full_url.lower() for indicator in path_indicators)):
            links.add(full_url)
        if len(links) >= max_links:
            break
    print(f"Extracted {len(links)} unique links matching criteria.")
    return list(links)[:max_links]

def analyze_article(text):
    """Run NER and sentiment analysis on text (with fallback)."""
    if ner_pipeline is None or sentiment_pipeline is None:
        return {"entities": "Analysis unavailable - install transformers", "sentiment": "N/A", "sentiment_score": 0.0}
    try:
        # NER: Limit to first 1400 chars
        ner_text = text[:1400]
        ents = ner_pipeline(ner_text)
        entities = " | ".join(set([f"{e['entity_group']}: {e['word']}" for e in ents if e['score'] > 0.8]))
        entities = entities or "None"
        # Sentiment: Limit to first 512 chars
        sent_text = text[:512]
        sent = sentiment_pipeline(sent_text)[0]
        sentiment_label = sent['label'].capitalize()
        sentiment_score = round(sent['score'], 3)
        return {"entities": entities, "sentiment": sentiment_label, "sentiment_score": sentiment_score}
    except Exception as e:
        print(f"Analysis error: {e}")
        return {"entities": "Error", "sentiment": "N/A", "sentiment_score": 0.0}

# -------------------------- MAIN SCRAPER FUNCTION --------------------------
def main_scraper(site_urls=['https://www.the-star.co.ke/'], max_articles=2):  # Reduced for testing
    """Scrape articles from sites, parse, analyze, and return DataFrame."""
    data = []
    # Selenium setup
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => false});")
        print("Chrome driver initialized in headless mode.")
    except Exception as e:
        print(f"Failed to init Chrome: {e}. Install Chrome and run 'pip install selenium webdriver-manager'.")
        return pd.DataFrame()

    for site in site_urls:
        print(f"\n--- Loading site: {site} ---")
        try:
            driver.get(site)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located(("tag name", "body")))
            time.sleep(3)  # Shorter for testing
            html = driver.page_source
            links = extract_article_links(html, site, max_articles)
            for url in links:
                try:
                    print(f"Fetching article: {url}")
                    article = Article(url)
                    article.download()
                    time.sleep(2)
                    article.parse()
                    # Skip old articles
                    if article.publish_date:
                        age_days = (datetime.now() - article.publish_date).days
                        if age_days > 30:
                            print(f"Skipping old article (age: {age_days} days)")
                            continue
                    text = article.text.strip()
                    if len(text) < 150:
                        print("Skipping short article (<150 chars)")
                        continue
                    # Categorize
                    category = categorize_article(text)
                    # Analyze
                    analysis = analyze_article(text)
                    # Prepare date
                    pub_date = article.publish_date.date() if article.publish_date else None
                    data.append({
                        "site_url": site,
                        "article_url": url,
                        "title": article.title.strip() if article.title else "No title",
                        "publish_date": pub_date,
                        "keyword_category": category,
                        "summary_snippet": (text[:200] + "...") if len(text) > 200 else text,
                        "full_text": text,
                        "entities": analysis["entities"],
                        "sentiment": analysis["sentiment"],
                        "sentiment_score": analysis["sentiment_score"]
                    })
                    print(f"Added analyzed article: {article.title[:50]}... (Category: {category})")
                except Exception as e:
                    print(f"Error processing {url}: {e}")
                    continue
        except Exception as e:
            print(f"Failed to load site {site}: {e}")
            continue

    if driver:
        driver.quit()
    print(f"\nScraping complete. Collected {len(data)} articles.")
    return pd.DataFrame(data) if data else pd.DataFrame()

# -------------------------- UPLOAD TO SUPABASE --------------------------
def upload_to_supabase(df):
    """Upsert DataFrame to Supabase 'scraped_articles' table on article_url."""
    if df.empty:
        print("No data to upload.")
        return
    if supabase is None:
        print("Supabase not initialized. Skipping upload.")
        return

    # Clean publish_date: Convert to datetime, then to date, then to ISO string for JSON serialization
    df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce').dt.date
    df['publish_date'] = df['publish_date'].apply(lambda x: x.isoformat() if x else None)

    # Ensure column order matches table
    cols = ["site_url", "article_url", "title", "publish_date", "keyword_category",
            "summary_snippet", "full_text", "entities", "sentiment", "sentiment_score"]
    final_df = df[cols].copy()
    records = final_df.to_dict(orient="records")

    # Save local CSV
    final_df.to_csv("scraped_articles.csv", index=False, encoding="utf-8")
    print(f"Saved local backup: scraped_articles.csv ({len(final_df)} articles)")

    # Upsert
    try:
        response = supabase.table("scraped_articles").upsert(records, on_conflict="article_url").execute()
        inserted_count = len(response.data) if response.data else 0
        print(f"SUCCESS: Uploaded {len(records)} records to 'scraped_articles' table.")
        if inserted_count == 0:
            print("No new inserts—check for duplicates or errors in response.")
        else:
            print("Data saved! Refresh Supabase Table Editor to see it.")
    except Exception as e:
        print(f"Supabase upload failed: {e}")
        print("Troubleshoot: Table exists? Columns match? Service key has INSERT perms?")

# -------------------------- MAIN EXECUTION --------------------------
if __name__ == "__main__":
    print("=== Article Scraper & Analyzer Starting (Target: scraped_articles table) ===")
    print(f"Current date/time: {datetime.now()}")
    # Run scraper
    raw_df = main_scraper()
    if raw_df.empty:
        print("No articles scraped. Check: Site up? Paths in extract_article_links? Increase max_articles?")
    else:
        print(f"\nUploading {len(raw_df)} analyzed articles to Supabase...")
        upload_to_supabase(raw_df)
    print("\n=== All done! Check CSV file and Supabase 'scraped_articles' table. ===")
