# api/display.py
import os
import json
from supabase import create_client, Client
import pandas as pd  # Optional: for better table formatting

# Set up your Supabase credentials (env vars from Vercel)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Validate env vars (optional: for debugging)
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY env vars")

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Table name based on your schema
TABLE_NAME = 'scraped_articles'

def handler(request):
    # Fetch all rows from the table (add .limit(50) if too many rows)
    response = supabase.table(TABLE_NAME).select('*').execute()

    if response.data:
        # Prepare table data
        try:
            df = pd.DataFrame(response.data)
            table_data = df.to_dict('records')  # Convert to list of dicts for JSON
        except ImportError:
            # Fallback: manual list of dicts
            table_data = response.data

        # Return success response with JSON body
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'  # For CORS if calling from frontend
            },
            'body': json.dumps({
                'message': 'Scraped Articles Table Data:',
                'data': table_data,
                'total_rows': len(response.data),
                'headers': list(table_data[0].keys()) if table_data else []
            })
        }
    else:
        # Error response
        error_msg = "No data found in the table or an error occurred."
        if hasattr(response, 'error') and response.error:
            error_msg += f" Details: {response.error}"
        
        return {
            'statusCode': 404,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': error_msg
            })
        }
