import pandas as pd
import numpy as np
from clickhouse_driver import Client
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import scipy
import torch
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CLICKHOUSE_HOST = ''
CLICKHOSE_USER = ''
CLICKHOSE_PWD = ''

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


def calc_sentiment(text: str, tokenizer: AutoTokenizer, model: AutoModelForSequenceClassification, model_name: str) -> tuple:
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
    with torch.no_grad():
        inputs = tokenizer(
            text, return_tensors='pt', padding=True, truncation=True, max_length=1024
        )
        outputs = model(**inputs)

        # convert logits to probabilities
        values = scipy.special.softmax(outputs.logits.numpy().squeeze())

        # initialize result, check none result in the upstream functions
        result = None

        if model_name == 'finbert':
            labels = [*model.config.id2label.values()]
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


def finbert_sentiment_fill_df(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
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
        copy_df[text_col].apply(lambda x: calc_sentiment(x, tokenizer_finbert, model_finbert, model_name='finbert')).apply(pd.Series)
    )
    return copy_df




#===================================
#  BERT Base SST-2 (General Purpose)
#===================================
def sst2_sentiment_fill_df(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    tokenizer_sst2 = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-SST-2')
    model_sst2 = AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-SST-2')

    copy_df = df.copy()
    copy_df[['sst2_pos', 'sst2_neg', 'sst2_stmt_score', 'sst2_stmt_label']] = (
        copy_df[text_col].apply(lambda x: calc_sentiment(x, tokenizer_sst2, model_sst2, model_name='sst2')).apply(pd.Series)
    )
    return copy_df


#=========================================================
# DeBERTa-v3 (State-of-the-Art) higher accuracy but larger
#=========================================================
tokenizer_deberta = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
model_deberta = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base')

def deberta_sentiment_fill_df(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    tokenizer_deberta = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    model_deberta = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base')

    copy_df = df.copy()
    copy_df[['deberta_pos', 'deberta_neg', 'deberta_stmt_score', 'deberta_stmt_label']] = (
        copy_df[text_col].apply(lambda x: calc_sentiment(x, tokenizer_deberta, model_deberta, model_name='deberta')).apply(pd.Series)
    )
    return copy_df


#=========================================================
# Ollama
#=========================================================


def process_news_data(data: pd.DataFrame):
    ticker_data = pd.read_csv(os.path.join(DATA_PATH, 'supported_tickers.csv'))
    ticker_list = (
    ticker_data[(
        ticker_data['exchange'].isin(['NYSE', 'NASDAQ', 'CSE', 'AMEX'])
        )]['ticker'].unique())
    data = data[
        (data['tickers_all'] != '') &
        (data['description'] != '') &
        (data['title'] != '')].copy()
    data['tickers_all'] = data['tickers_all'].str.upper().str.split('/')
    data = data[data['tickers_all'].apply(lambda x: set(x).issubset(set(ticker_list)))]
    return data

def get_news_data(start_date: datetime, end_date: datetime):
    client = Client(
        CLICKHOUSE_HOST,
        user=CLICKHOSE_USER,
        password=CLICKHOSE_PWD,
        database='tiingo'
    )

    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    sql = f"""
    select distinct(id), description, tickers_all, title, parseDateTimeBestEffort(publishedDate) AS date 
    from news
    where date >= '{start_date}'
    and date < '{end_date}'
    limit 5
    """

    res = fetch_database(client, sql)
    news_data = pd.DataFrame(res, columns=['id', 'description', 'tickers_all', 'title', 'date'])
    return news_data

def main():
    
    start_date = datetime(2016, 1, 1)
    end_date = datetime(2016, 2, 1)
    new_data = get_news_data(start_date, end_date)
    process_data = process_news_data(new_data)
    
    if len(process_data) != 0:
        test_df = finbert_sentiment_fill_df(process_data, text_col='title')
        print(test_df)


if __name__ == '__main__':
    main()