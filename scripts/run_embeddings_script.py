from mistralai import Mistral
import time
from tqdm import tqdm
import pandas as pd
import argparse
from clickhouse_driver import Client

def run_mistral_embedding(df: pd.DataFrame, api_key, model: str = 'mistral-embed', batch_size: int = 10) -> None:
    '''
    Given the news dataframe, append the new column with mistral embeddings values (list cast to str, for now)
    Be reminded that this function is inplace, modifying the df directly.
    '''

    mistral_client = Mistral(api_key=api_key)

    # Initialize new column
    df['mistral_embedding'] = None

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
            df.loc[start_idx + i, 'mistral_embedding'] = str(embedding.embedding)

        # request rate is limited, mistral is annoying
        time.sleep(2)

def fetch_news_data(host: str, user: str, password: str) -> pd.DataFrame:
    
    client = Client(host=host, user=user, password=password)

    # Query data
    query = """
    SELECT  distinct(id) as id,
            description,
            tickers_all,
            title,
            parseDateTimeBestEffort(publishedDate) AS date 
    FROM tiingo.news
    WHERE ticker = 'btc'
    """

    # Fetch the result directly as a pandas DataFrame
    return client.query_dataframe(query)

def main():
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('-gihost', help="Database hostname")
    parser.add_argument('-u', '--user', help="Database username")
    parser.add_argument('-p', '--password', help="Database password")
    parser.add_argument('-k', '--api_key', help="Mistral API key")

    # Parse the arguments
    args = parser.parse_args()

    news_df = fetch_news_data(args.host, args.user, args.password)

    run_mistral_embedding(news_df, api_key=args.api_key, model='mistral-embed')

    # save the news_data_df to a parquet file
    news_df.to_parquet('news_data_embeddings.parquet', index=False)

if __name__ == '__main__':
    main()