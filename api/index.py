# api/index.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
# ... (import all your other deps: selenium, newspaper, etc.)
# Paste your full code here (main_scraper, helpers, analysis, Supabase upload)

app = FastAPI()

@app.get("/")
async def run_scraper():
    try:
        raw_df = main_scraper()
        if raw_df is None or len(raw_df) == 0:
            return JSONResponse({"error": "No articles scraped"}, status_code=404)
        
        # ... (your analysis and final_df code)
        
        # Upload to Supabase (keep as-is)
        # ...
        
        return {
            "status": "success",
            "articles_count": len(final_df),
            "data": final_df.to_dict(orient="records")  # Or summarize to avoid huge responses
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# For POST triggers (e.g., from a webhook), add @app.post("/")