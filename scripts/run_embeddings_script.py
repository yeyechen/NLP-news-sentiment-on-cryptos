from mistralai import Mistral
import time
from tqdm import tqdm
import pandas as pd
import argparse
from clickhouse_driver import Client
import numpy as np
import json
import os
from transformers import AutoModel

import requests
import time



JINA_MODEL = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

data_path = '../Mikey_data/'
if os.path.exists(data_path):
    os.chdir(data_path)
    print('Changed directory to data path:', os.getcwd())
else:
    print('Data path does not exist:', data_path)

with open('credentials.json', 'r') as file:
    credentials = json.load(file)

DB_SERVER_NAME = credentials['clickhouse']['server_name']
DB_USERNAME = credentials['clickhouse']['username']
DB_PASSWORD = credentials['clickhouse']['password']

MISTRAL_API = credentials['api_key']['mistral']
HUGGINGFACE_API = credentials['api_key']['huggingface_daniel']


def run_jina_embedding(df: pd.DataFrame, text_column: str = "description", batch_size: int = 50, truncate_dim: int = 1024) -> pd.DataFrame:
    """
    Compute embeddings for texts in a DataFrame using the locally installed Jina AI model.
    The DataFrame is processed in batches, and results are assigned slice-wise.
    
    Args:
        df (pd.DataFrame): DataFrame containing the news data.
        text_column (str): Column name containing the text to embed (default "description").
        batch_size (int): Number of rows to process per batch (default 50).
        truncate_dim (int): Truncation dimension to be used in model.encode (default 1024).
    
    Returns:
        pd.DataFrame: The original DataFrame with a new column "jina_embeddings"
                      storing the embedding vectors.
    """
    # Initialize a new column for embeddings
    df["jina_embeddings"] = None
    
    # Process DataFrame in batches
    for start_idx in tqdm(range(0, len(df), batch_size)):
        end_idx = min(start_idx + batch_size, len(df))
        batch_texts = df.loc[start_idx:end_idx-1, text_column].tolist()
        
        # Compute embeddings for the batch
        embedding_output = JINA_MODEL.encode(batch_texts, truncate_dim=truncate_dim)
        
        for i, embedding in enumerate(embedding_output):
            df.loc[start_idx + i, "jina_embeddings"] = str(embedding)
    
    return df

def run_miniLM_embedding(df: pd.DataFrame, api_token=HUGGINGFACE_API, batch_size=10, max_retries=3) -> None:
    """
    Process text embeddings in batches using the sentence-transformers/all-MiniLM-L6-v2 model.
    
    Args:
        df: pandas DataFrame containing the text data
        api_token: Hugging Face API token
        batch_size: number of texts to process in each batch
        max_retries: maximum number of retry attempts per batch
    """

    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {api_token}"}

    # Initialize new column
    df['miniLM_embeddings'] = None

    # Process in batches
    for start_idx in tqdm(range(0, len(df), batch_size)):
        end_idx = min(start_idx + batch_size, len(df))
        batch_texts = df.loc[start_idx:end_idx - 1, 'description'].tolist()

        # Try to get embeddings with retries
        for attempt in range(max_retries):
            response = requests.post(
                api_url, 
                headers=headers, 
                json={
                    "inputs": batch_texts,
                    "options": {
                        "wait_for_model": True,
                        "use_cache": False 
                    }
                }
            )
            
            if response.status_code == 200:
                embeddings = response.json()
                # Store embeddings in DataFrame
                for i, embedding in enumerate(embeddings):
                    df.loc[start_idx + i, 'miniLM_embeddings'] = str(embedding)
            elif response.status_code == 503 and attempt < max_retries - 1:
                # Model loading - wait with exponential backoff
                wait_time = 2 ** attempt
                print(f"Model loading, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Embedding failed for batch {start_idx}-{end_idx}: {response.text}")
                continue


def run_mistral_embedding(df: pd.DataFrame, api_key=MISTRAL_API, model: str = 'mistral-embed', batch_size: int = 50) -> None:
    '''
    Given the news dataframe, append the new column with mistral embeddings values (list cast to str, for now)
    Be reminded that this function is inplace, modifying the df directly.
    '''

    mistral_client = Mistral(api_key=api_key)

    # Initialize new column
    df['mistral_embeddings'] = None

    # Process in batches
    for start_idx in tqdm(range(0, len(df), batch_size)):
        end_idx = min(start_idx + batch_size, len(df))
        batch_texts = df.loc[start_idx:end_idx - 1, 'description'].tolist()

        embeddings_batch_response = mistral_client.embeddings.create(
            model=model,
            inputs=batch_texts,
        )

        # Write result embeddings values to df
        for i, embedding in enumerate(embeddings_batch_response.data):
            df.loc[start_idx + i, 'mistral_embeddings'] = str(embedding.embedding)

        # request rate is limited, mistral is annoying
        time.sleep(2)


def fetch_news_data(host: str, user: str, password: str) -> pd.DataFrame:
    
    client = Client(host=host, user=user, password=password)

    # Query data
    query = """
        SELECT  distinct(id) as id,
                tags,
                description,
                ticker,
                tickers_all,
                title,
                parseDateTimeBestEffort(publishedDate) AS date 
        FROM tiingo.news
        WHERE (ticker in ('btc', 'eth', 'doge', 'sol') OR LOWER(tags) like '%crypto%')
        AND LENGTH(description) >= 12
        AND LENGTH(description) <= 8192
        AND date >= '2018-01-01'
        AND description != ''
        AND description != ' '
    """

    query_btc = """
        SELECT  distinct(id) as id,
                tags,
                description,
                ticker,
                tickers_all,
                title,
                parseDateTimeBestEffort(publishedDate) AS date 
        FROM tiingo.news
        WHERE ticker = 'btc'
        AND LENGTH(description) >= 12
        AND date >= '2018-01-01'
        AND description != ''
        AND description != ' '
    """

    # Fetch the result directly as a pandas DataFrame
    return client.query_dataframe(query_btc)


def fetch_additional_news_data(filename: str)-> pd.DataFrame:
    df = pd.read_csv(filename)
    df.rename(columns={'title': 'description', 'newsDatetime': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # df = df[df['date'] >= '2024-01-01']
    df = df.dropna(subset=['description'])
    return df

def main():
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('--host', help="Database hostname")
    parser.add_argument('-u', '--user', help="Database username")
    parser.add_argument('-p', '--password', help="Database password")
    parser.add_argument('-k', '--api_key', help="Mistral API key")
    parser.add_argument('--model', help="Model name", default='mistral-embed')

    # Parse the arguments
    args = parser.parse_args()


    # news_df = fetch_news_data(db_server_name, db_username, db_password)
    news_df = fetch_additional_news_data('news_currencies_source_joinedResult.csv')
    news_df = news_df.drop_duplicates(subset=['id']).reset_index(drop=True)

    model_func = None
    col_name = None
    if args.model == 'jina':
        model_func = run_jina_embedding
        col_name = 'jina_embeddings'
    elif args.model == 'mistral':
        model_func = run_mistral_embedding
        col_name = 'mistral_embeddings'
    elif args.model == 'miniLM':
        model_func = run_miniLM_embedding
        col_name = 'miniLM_embeddings'

    # Create a new folder if it does not exist
    output_folder = f"{args.model}_additional_data_embeddings"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")
    else:
        print(f"Folder already exists: {output_folder}")

    os.chdir(output_folder)
    print('Changed directory to output folder:', os.getcwd())
    
    # Group the dataframe by year and month
    news_df['year_month'] = news_df['date'].dt.strftime('%Y-%m')
    
    # Get unique year-month combinations
    month_groups = sorted(news_df['year_month'].unique())
    
    print(f"Processing {len(month_groups)} monthly groups")
    
    # Process each month separately
    for ym in month_groups:
        print(f"Processing month: {ym}")
        
        # Filter data for this month
        month_df = news_df[news_df['year_month'] == ym].copy()
        month_df.reset_index(drop=True, inplace=True)
        
        # Skip if no data
        if len(month_df) == 0:
            print(f"No data for month {ym}, skipping...")
            continue
            
        print(f"Found {len(month_df)} records for {ym}")
        
        # Generate embeddings for this month's data
        model_func(month_df)
        
        # Save to a month-specific file
        output_filename = f'{col_name}_additional_news{ym}.parquet'
        month_df[['id', col_name]].to_parquet(output_filename, index=False)
        print(f"Saved embeddings to {output_filename}")
    
    print("Done")

if __name__ == '__main__':
    main()