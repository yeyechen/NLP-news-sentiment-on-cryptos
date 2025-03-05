import scipy
import torch
import os
import warnings
import pandas as pd
import numpy as np
import argparse
from typing import List
from datetime import datetime
from clickhouse_driver import Client
import torch.nn.functional as F


from config import DATA_PATH, CLICKHOUSE_HOST
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings('ignore')
print('data path: ', DATA_PATH)


print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')  # Should be True on servers
print(f'CUDA version: {torch.version.cuda}')  # Should show 12.3 on servers

#========
# FinBERT
#========

def fetch_database(client, sql):
    res = client.execute(sql)
    client.disconnect()
    return res


def calc_sentiment(text: str, 
                   tokenizer: AutoTokenizer, 
                   model: AutoModelForSequenceClassification, 
                   model_name: str,
                   device: str) -> tuple:
    '''
    Analyzes the sentiment of a given text using the given tokenizer and model.

    Args:
        text (str): The text to analyze.
        tokenizer: The tokenizer to use.
        model: The model to use. Model should be a classification model.
        model_name (str): Name of the model - different model different output format.
    Returns:
        tuple: A tuple containing:
            - float: Probability of positive sentiment.
            - float: Probability of negative sentiment.
            - float: Probability of neutral sentiment.
            - float: Sentiment score (positive - negative).
            - str: Sentiment label with the highest probability.
    '''

    model.to(device)
    model.eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    with torch.no_grad():
        inputs = tokenizer(
            text, return_tensors='pt', padding=True, truncation=True, max_length=1024
        )
        inputs.to(device)
        outputs = model(**inputs)

        # convert logits to probabilities
        values = scipy.special.softmax(outputs.logits.cpu().numpy().squeeze())

        # initialize result, check none result in the upstream functions
        result = None

        if model_name == 'finbert':
            labels = [*model.module.config.id2label.values()] if isinstance(model, torch.nn.DataParallel) else [*model.config.id2label.values()]
            result = (
                values[0], # positive
                values[1], # negative
                values[2], # neutral
                values[0] - values[1], # sentiment score
                labels[np.argmax(values)], # sentiment label (max probability label)
            )

        if model_name == 'sst2' or model_name == 'deberta':
            label_map = {
                0: 'negative',
                1: 'positive'
            }
            label_idx = np.argmax(values)
            result = (
                values[1], # positive
                values[0], # negative
                values[1] - values[0], # sentiment score
                label_map[label_idx], # sentiment label (max probability label)
            )

        return result


def finbert_sentiment_fill_df(df: pd.DataFrame, text_col: str, device:str) -> pd.DataFrame:
    '''
    This function takes a DataFrame and a column name containing text data, and returns a new DataFrame
    with additional columns for FinBERT sentiment analysis scores and labels.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing text data.
    text_col (str): The name of the column in the DataFrame that contains the text data to be analyzed.

    Returns:
    pd.DataFrame: A new DataFrame with the original data and additional columns:
        - finbert_pos: Probability of positive sentiment.
        - finbert_neg: Probability of negative sentiment.
        - finbert_neu: Probability of neutral sentiment.
        - finbert_score: Sentiment score calculated as (positive - negative).
        - finbert_sentiment: Sentiment label with the highest probability.
    '''
    tokenizer_finbert = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model_finbert = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

    copy_df = df.copy()
    copy_df[['finbert_pos', 'finbert_neg', 'finbert_neu', 'finbert_stmt_score', 'finbert_stmt_label']] = (
        copy_df[text_col].apply(lambda x: calc_sentiment(x, tokenizer_finbert, model_finbert, model_name='finbert', device=device)).apply(pd.Series)
    )
    return copy_df




#===================================
#  BERT Base SST-2 (General Purpose)
#===================================
def sst2_sentiment_fill_df(df: pd.DataFrame, text_col: str, device: str) -> pd.DataFrame:
    tokenizer_sst2 = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-SST-2')
    model_sst2 = AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-SST-2')

    copy_df = df.copy()
    copy_df[['sst2_pos', 'sst2_neg', 'sst2_stmt_score', 'sst2_stmt_label']] = (
        copy_df[text_col].apply(lambda x: calc_sentiment(x, tokenizer_sst2, model_sst2, model_name='sst2', device=device)).apply(pd.Series)
    )
    return copy_df


#=========================================================
# DeBERTa-v3 (State-of-the-Art) higher accuracy but larger
#=========================================================
tokenizer_deberta = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
model_deberta = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base')

def deberta_sentiment_fill_df(df: pd.DataFrame, text_col: str, device: str) -> pd.DataFrame:
    tokenizer_deberta = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    model_deberta = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base')

    copy_df = df.copy()
    copy_df[['deberta_pos', 'deberta_neg', 'deberta_stmt_score', 'deberta_stmt_label']] = (
        copy_df[text_col].apply(lambda x: calc_sentiment(x, tokenizer_deberta, model_deberta, model_name='deberta', device=device)).apply(pd.Series)
    )
    return copy_df

def calc_sentiment_roberta(text: str, tokenizer, model, device):
    # Tokenize the input text; padding ensures consistency for short texts.
    model.to(device)
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs.to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    # Compute probabilities with softmax
    probabilities = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    
    # For the "cardiffnlp/twitter-xlm-roberta-base-sentiment" model,
    # the label order is: 0 -> negative, 1 -> neutral, 2 -> positive.
    neg_prob = round(float(probabilities[0]), 6)
    neu_prob = round(float(probabilities[1]), 6)
    pos_prob = round(float(probabilities[2]), 6)
    
    # Calculate the sentiment score as positive minus negative.
    score = round(pos_prob - neg_prob, 6)
    
    # Get the label with the highest probability.
    label = ['negative', 'neutral', 'positive'][int(probabilities.argmax())]
    
    return pd.Series([pos_prob, neg_prob, neu_prob, score, label])

def roberta_sentiment_fill_df(df: pd.DataFrame, text_col: str, device:str) -> pd.DataFrame:
    '''
    This function takes a DataFrame and a column name containing text data, and returns a new DataFrame
    with additional columns for Roberta sentiment analysis scores and labels.

    Parameters:
      df (pd.DataFrame): The input DataFrame containing text data.
      text_col (str): The name of the column in the DataFrame that contains the text data to be analyzed.

    Returns:
      pd.DataFrame: A new DataFrame with the original data and additional columns:
          - roberta_pos: Probability of positive sentiment.
          - roberta_neg: Probability of negative sentiment.
          - roberta_neu: Probability of neutral sentiment.
          - roberta_stmt_score: Sentiment score calculated as (positive - negative).
          - roberta_stmt_label: Sentiment label with the highest probability.
    '''
    # Load the tokenizer and model.
    tokenizer_roberta = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
    model_roberta = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
    
    copy_df = df.copy()
    # Apply the helper function to the text column.
    copy_df[['roberta_pos', 'roberta_neg', 'roberta_neu', 'roberta_stmt_score', 'roberta_stmt_label']] = (
        copy_df[text_col]
        .apply(lambda x: calc_sentiment_roberta(x, tokenizer_roberta, model_roberta, device=device))
        .apply(pd.Series)
    )
    return copy_df


#=========================================================
# Ollama
#=========================================================

def process_news_data(data: pd.DataFrame, ticker_list: List[str]) -> pd.DataFrame:
    # Precompute the set for fast membership testing
    ticker_set = set(ticker_list)
    
    # Filter out rows where any key column is empty using vectorized string operations
    mask = (
        (data['tickers_all'].astype(str).str.strip() != '' )&
        (data['description'].astype(str).str.strip() != '') &
        (data['title'].astype(str).str.strip() != '')
    )
    data = data[mask].copy()
    
    # Convert tickers_all to uppercase and split on '/'
    data['tickers_all'] = data['tickers_all'].str.upper().str.split('/')
    
    # Cache the subset check result for repeated ticker lists
    cache = {}
    def is_valid_tickers(tickers):
        key = tuple(tickers)  # Use tuple as a hashable cache key
        if key not in cache:
            cache[key] = set(tickers).issubset(ticker_set)
        return cache[key]
    
    # Apply the cached subset check. List comprehensions are often faster than apply.
    valid_mask = [is_valid_tickers(tickers) for tickers in data['tickers_all']]
    data = data[valid_mask]
    
    return data

def get_news_data(user_name:str, pwd:str, 
                  start_date: datetime, end_date: datetime):
    client = Client(
        CLICKHOUSE_HOST,
        user=user_name,
        password=pwd,
        database='tiingo'
    )

    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    sql = f"""
    select distinct(id), description, tickers_all, title, parseDateTimeBestEffort(publishedDate) AS date 
    from news
    where date >= '{start_date}'
    and date < '{end_date}'
    """

    res = fetch_database(client, sql)
    news_data = pd.DataFrame(res, columns=['id', 'description', 'tickers_all', 'title', 'date'])
    return news_data

def run_model(data, model, device):
    if model == 'finbert':
        results = finbert_sentiment_fill_df(data, text_col='title', device=device)
        results = results[['id', 'finbert_pos', 'finbert_neg', 'finbert_neu', 'finbert_stmt_score', 'finbert_stmt_label']]
        return results
    elif model == 'roberta':
        results =  roberta_sentiment_fill_df(data, text_col='title', device=device)
        results = results[['id', 'roberta_pos', 'roberta_neg', 'roberta_neu', 'roberta_stmt_score', 'roberta_stmt_label']]
        return results
    
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('--model', help="Model")
    parser.add_argument('--device', help='cuda device')
    parser.add_argument('-v', '--verbose', default=True, action='store_true', help="Increase output verbosity")
    parser.add_argument('-u', '--user', help="Database user name")
    parser.add_argument('-p', '--password', help="Database user password")

    # Parse the arguments
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    current_date = datetime(2019, 10, 11)
    end_date = datetime(2023, 1, 1)
    ticker_data = pd.read_csv(os.path.join(DATA_PATH, 'supported_tickers.csv'))
    ticker_list = (
    ticker_data[(
        ticker_data['exchange'].isin(['NYSE', 'NASDAQ', 'CSE', 'AMEX'])
        )]['ticker'].unique())
    
    while current_date < end_date:
        # Calculate start and end of month
        month_start = current_date
        if current_date.month == 12:
            month_end = datetime(current_date.year + 1, 1, 1)
        else:
            month_end = datetime(current_date.year, current_date.month + 1, 1)
        
        if args.verbose:
            print(f'Processing {month_start.strftime("%Y-%m")}...')
        
        new_data = get_news_data(user_name=args.user,
                                 pwd=args.password,
                                 start_date=month_start, 
                                 end_date=month_end)
        if args.verbose:
            print('got new data...')
        process_data = process_news_data(new_data, ticker_list)
        if args.verbose:
            print('processed new data, length: ', len(process_data))
        
        if len(process_data) != 0:
            test_df = run_model(process_data, args.model, device)
            filename = f'sentiment_{month_start.strftime("%Y_%m")}.parquet'
            test_df.to_parquet(os.path.join(DATA_PATH, f'{args.model}/'+filename))
            if args.verbose:
                print(f'Saved sentiment results for {month_start.strftime("%Y-%m")} to {filename}')
        
        # Move to next month
        current_date = month_end
    
if __name__ == '__main__':
    main()
