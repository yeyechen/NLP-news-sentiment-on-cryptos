from mistralai import Mistral
import time
from tqdm import tqdm
import pandas as pd
import argparse
from clickhouse_driver import Client
import numpy as np
import json
import os

import requests
import time

def run_miniLM_embedding(df: pd.DataFrame, api_token: str, batch_size=10, max_retries=3) -> None:
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


def run_mistral_embedding(df: pd.DataFrame, api_key, model: str = 'mistral-embed', batch_size: int = 50) -> None:
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
        AND date >= '2018-01-01'
        AND description != ''
        AND description != ' '
    """

    # Fetch the result directly as a pandas DataFrame
    return client.query_dataframe(query_btc)

def main():
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('--host', help="Database hostname")
    parser.add_argument('-u', '--user', help="Database username")
    parser.add_argument('-p', '--password', help="Database password")
    parser.add_argument('-k', '--api_key', help="Mistral API key")
    parser.add_argument('--model', help="Model name", default='mistral-embed')

    data_path = '../Mikey_data/'
    if os.path.exists(data_path):
        os.chdir(data_path)
        print('Changed directory to data path:', os.getcwd())
    else:
        print('Data path does not exist:', data_path)

    with open('credentials.json', 'r') as file:
        credentials = json.load(file)

    db_server_name = credentials['clickhouse']['server_name']
    db_username = credentials['clickhouse']['username']
    db_password = credentials['clickhouse']['password']

    mistral_api = credentials['api_key']['mistral']
    huggingface_api = credentials['api_key']['huggingface2']

    # Parse the arguments
    args = parser.parse_args()

    news_df = fetch_news_data(db_server_name, db_username, db_password)
    news_df = news_df.drop_duplicates(subset=['id']).reset_index(drop=True)

    if args.model == 'mistral':
        run_mistral_embedding(news_df, api_key=mistral_api, model='mistral-embed')
        news_df[['id', 'mistral_embeddings']].to_parquet('mistral_embeddings.parquet', index=False)

    elif args.model == 'miniLM':
        run_miniLM_embedding(news_df, api_token=huggingface_api)
        news_df[['id', 'miniLM_embeddings']].to_parquet('miniLM_embeddings_btc_only.parquet', index=False)

if __name__ == '__main__':
    main()