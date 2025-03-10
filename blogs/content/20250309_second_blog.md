---
Title: Second Blog Post (by Group "Non JaneStreet")
Date: 2025-03-09 23:59
Category: Reflective Report
Tags: Group Non Jane Street
---

## Introduction

In our second blog post, we dive deeper into our financial sentiment analysis project, building on the foundation laid in our first blog post. Over the past two weeks, our group has focused on aggregating and clustering sentiment scores derived from financial news articles using FinBERT and RoBERTa models. We processed large datasets to create daily sentiment clusters, analyzed sector-specific sentiment trends, and prepared the data for potential time-series analysis to uncover relationships with market movements. This post details our methodology, challenges, and key findings, showcasing how we transformed raw sentiment data into structured, actionable insights.

## Data Preparation and Sentiment Aggregation

Our journey began with the raw sentiment data generated from FinBERT and RoBERTa models, which we had previously applied to financial news articles stored in a ClickHouse database (as discussed in our first blog). The datasets contained sentiment scores (positive, negative, neutral) for news articles linked to specific stock tickers, along with metadata like publication dates and sector classifications. Our goal was to aggregate these sentiment scores into meaningful daily clusters, both at a market-wide level and by sector, to enable further time-series analysis.

### Loading and Filtering Data

We started by loading the processed sentiment data into Pandas DataFrames. The first notebook focused on RoBERTa sentiment scores, while the second used FinBERT scores. Both datasets were filtered to include only records from August 17, 2017, to September 6, 2022, aligning with our target analysis period. Below is a snippet of the filtering process:

```python
df = df[df['date'] >= '2017-08-17']
```

Additionally, we used ticker count data (`sNv.pkl`) to understand the number of unique stocks mentioned per day and per sector, which helped us normalize sentiment scores later. This step ensured we accounted for varying news coverage across different stocks and sectors.

### Sentiment Clustering Methodology

To aggregate sentiment scores, we developed a custom `sentiment_cluster` function that addressed two key challenges:

- **Avoiding Double Counting:** Some stocks had multiple news articles on the same day, which could skew sentiment scores. We averaged sentiment scores for each stock on a given day to mitigate this.

- **Handling Missing News:** For stocks without news on a particular day, we assumed sentiment remained the same as the previous period, using forward-filling to propagate scores.

Here’s a simplified version of our clustering logic:

```python
def sentiment_cluster(df_, stock_counts, fin_col, freq='d', impact=3):
    """
    params:
       df_:
       stock_counts:
       fin_col:
       freq:
       impact:
    """
    """ps: 1. some stocks have many news during same period, we should avoid double counting."""
    df_ = df_.copy()
    if freq == 'h':
        freq_col = 'publishedDate'
        stock_counts.name = 'count'
        df_ = pd.merge(df_, stock_counts, left_on='date', right_index=True, how='left')
        stock_counts = df_[['count', freq_col]].drop_duplicates().set_index([freq_col])
    else:
        freq_col = 'date'
    # for the repeated news about the same stock at a certain period, we take the average.
    temp_senti_s = df_.groupby(['ticker', freq_col]).apply(lambda x: x[['finbert_pos', 'finbert_neg', 'finbert_neu']].mean(axis=0))
    # suppose the stocks have no news remain the same sentiment score as last period.
    temp_senti_freq = temp_senti_s.groupby(level=1).mean()
    temp_count_freq = temp_senti_s.groupby(level=1).apply(lambda x: x.shape[0])
    temp_count_freq.name = 'inner_count'
    temp_merge = pd.concat([stock_counts, temp_count_freq, temp_senti_freq], axis=1).sort_index().ffill()
    cluster_dict = {'finbert_pos': [], 'finbert_neg': [], 'finbert_neu': []}
    last_senti = {'finbert_pos': temp_merge.head(1)['finbert_pos'].values[0],
                  'finbert_neg': temp_merge.head(1)['finbert_neg'].values[0],
                  'finbert_neu': temp_merge.head(1)['finbert_neu'].values[0]}
    for ind, row in temp_merge.iterrows():
        for s in ['finbert_pos', 'finbert_neg', 'finbert_neu']:
            temp_put = (row[s] * impact * row['inner_count'] + last_senti[s] * (row['count'] - row['inner_count'])
                           ) / (impact * row['inner_count'] + row['count'] - row['inner_count'])
            
            cluster_dict[s].append(temp_put)
            last_senti[s] = temp_put
    return pd.DataFrame(data=cluster_dict, index=temp_merge.index)
```
The impact parameter (set to 3) gave more weight to stocks with news coverage, ensuring that sentiment scores reflected actual news-driven sentiment more heavily than assumed carryover values.

### Sector-Specific Clustering

We extended our clustering approach to analyze sentiment by sector using the GICS sector classifications (`gsector`) in our dataset. This allowed us to capture sector-specific trends, which are critical for understanding how broader market sentiment might differ across industries like technology, finance, or energy. The `sector_cluster` function iterated over each sector, applying the same clustering logic:

```python
def sector_cluster(df_, stock_counts_sector):
    sec_dict = {}
    for sector, group in df_.groupby('gsector'):
        tc = stock_counts_sector[stock_counts_sector.index.get_level_values(0) == sector].reset_index(level=0, drop=True)
        temp_d = sentiment_cluster(group, tc, 'd')
        sec_dict[sector] = temp_d
    return sec_dict
```

The output was a dictionary of DataFrames, each containing daily sentiment scores for a specific sector. We saved these results as pickled files (`roberta_sector_sentiment.pkl` and `finbert_sentiment_by_sector_dict.pkl`) for future analysis.

## Why Sentiment Clustering and Market Proxies?

Raw sentiment scores for individual news articles are useful but noisy—stocks often have multiple news articles per day, and sentiment can vary widely within short periods. To address this, we developed `finbert_cluster` to aggregate sentiment scores at a daily level, capturing the overall market mood while accounting for the volume of news. Additionally, we used `sentiment_proxy`—a dataset reflecting market fear and greed indices—to contextualize our sentiment data against broader market sentiment trends. Our goal was to test whether aggregated sentiment signals could predict or correlate with market movements, particularly in the volatile cryptocurrency space.

### Clustering Sentiment Scores with `finbert_cluster`

We started by loading our pre-processed FinBERT sentiment data (`final_m.pkl`), which contains sentiment scores (`finbert_pos`, `finbert_neg`, `finbert_neu`, and a composite `finbert` score) for news articles from 2017-08-17 onward . To make this data actionable, we implemented a custom function `sentiment_cluster` to aggregate sentiment scores by ticker and date. The function:

- Groups news by ticker and date (or hour, depending on the frequency parameter).
- Computes weighted averages of sentiment scores, factoring in the number of news articles per period.
- Handles missing data by propagating the last known sentiment, ensuring continuity in the time series.

The resulting `finbert_cluster` dataset provides daily aggregated sentiment scores, as shown below:

| Index | finbert_pos | finbert_neg | finbert_neu | dt         | close_value |
|-------|-------------|-------------|-------------|------------|-------------|
| 1667  | 0.231701    | 0.127091    | 0.641208    | 2017-08-17 | ...         |
| 1668  | 0.199946    | 0.157277    | 0.642777    | 2017-08-17 | ...         |
| ...   | ...         | ...         | ...         | ...        | ...         |
| 3115  | 0.274161    | 0.186414    | 0.539425    | 2022-09-06 | ...         |


Aggregating sentiment scores into `finbert_cluster` allowed us to smooth out the noise inherent in individual news articles and focus on broader trends. By weighting the sentiment scores by the number of articles (`inner_count`), we ensured that high-volume news days had a proportional impact on the overall sentiment signal. This step was crucial for downstream analysis, as it provided a stable time series we could merge with market data and use for causality testing and strategy development.

### Integrating with `sentiment_proxy` for Market Context

To contextualize our sentiment data, we integrated `finbert_cluster` with a market sentiment proxy (`sentiment_proxy`), which we sourced from a fear and greed index dataset (`alexey-formalmethods_fear_gr`). This dataset provides a daily measure of market sentiment, often used to gauge investor fear or greed. We merged `finbert_cluster` with `sentiment_proxy` on the date index, creating a unified dataset that combines news sentiment with market sentiment:

| Index | finbert_pos | finbert_neg | finbert_neu | dt         | close_value |
|-------|-------------|-------------|-------------|------------|-------------|
| 1667  | 0.231701    | 0.127091    | 0.641208    | 2017-08-17 | ...         |
| ...   | ...         | ...         | ...         | ...        | ...         |

We then performed Granger causality tests to explore whether our aggregated sentiment scores (`finbert_pos`) could predict movements in the market proxy. The results showed significant causality at certain lags (e.g., p=0.0023 at lag 2), suggesting a potential predictive relationship.


Merging `finbert_cluster` with `sentiment_proxy` allowed us to test whether news-derived sentiment aligns with or predicts broader market sentiment. The fear and greed index captures investor psychology, which often drives market movements. By combining it with our FinBERT-derived sentiment, we aimed to uncover whether news sentiment could serve as an early indicator of market shifts—a critical insight for trading strategies.

## Developing Sentiment-Driven Trading Strategies

With our aggregated sentiment data in hand, we turned to developing trading strategies to test its predictive power on cryptocurrency markets. We implemented two strategies: `sentiment_strategy` and `moving_average_strategy`.

### `sentiment_strategy`: Threshold-Based Trading

The `sentiment_strategy` function uses sentiment scores as a signal to buy or sell cryptocurrencies. It takes a sentiment factor (e.g., rolling mean of `finbert_pos`), a cryptocurrency dataset (e.g., BTC, ETH, DOGE, SOL), and buy/sell thresholds (e.g., 0.75). The logic is as follows:

- If the sentiment score falls below a dynamic buy threshold (based on a rolling window), buy (signal=1).
- If the sentiment score exceeds a sell threshold, sell (signal=0).
- Forward-fill positions with a limit to avoid excessive holding periods (`limited_ffill`).

We tested this strategy on BTC, DOGE, SOL, and ETH, calculating cumulative returns and visualizing results.

放图片

The `sentiment_strategy` tests whether sentiment extremes (very positive or very negative) can predict short-term price movements in cryptocurrencies. Cryptocurrencies are highly sentiment-driven, making them an ideal testbed for our news-based sentiment signals. By setting dynamic thresholds, we aimed to adapt to changing market conditions and avoid overfitting to static rules.

### `moving_average_strategy`: Trend-Based Trading

The `moving_average_strategy` applies a classic moving average crossover approach to sentiment scores. It calculates short-term (e.g., 5-day) and long-term (e.g., 20-day) moving averages of the sentiment factor and generates signals:

- Buy (signal=1) when the short-term MA crosses below the long-term MA (indicating a potential reversal).
- Sell (signal=0) when the short-term MA crosses above the long-term MA.
- Compute cumulative returns and plot the results.

We tested this on BTC data, using a factor derived from sector-specific sentiment.

The moving average strategy leverages trends in sentiment rather than absolute levels, aiming to capture momentum shifts. This complements the `sentiment_strategy` by providing a different perspective on sentiment dynamics. We chose this approach because moving averages are widely used in technical analysis, and applying them to sentiment data allowed us to test whether sentiment trends behave similarly to price trends.

## Results and Observations

### Sentiment Clustering and Market Proxy Integration

- The `finbert_cluster` dataset successfully aggregated sentiment scores, reducing noise and providing a stable time series for analysis.
- Granger causality tests indicated a statistically significant relationship between `finbert_pos` and the market proxy at certain lags, suggesting that news sentiment may have predictive power over market sentiment.
- Visualizations of `finbert_cluster` and `sentiment_proxy` (20-day rolling means) revealed periods of alignment and divergence, providing qualitative insights into sentiment dynamics.

### Trading Strategies

- The `sentiment_strategy` produced varied results across cryptocurrencies. For BTC, cumulative returns showed periods of outperformance, but volatility remained high. Similar patterns emerged for DOGE, SOL, and ETH.
- The `moving_average_strategy` captured some sentiment trends but underperformed in highly volatile periods, suggesting that sentiment trends may lag price movements in fast-moving markets like crypto.



## Challenges

- **Data Volume and Memory Management**: Processing millions of news records (e.g., 13.7 million rows in the FinBERT dataset) required careful memory management. We used `gc.collect()` frequently and processed data in chunks where possible.

- **Temporal Gaps**: Some days had sparse news coverage for certain stocks or sectors, necessitating assumptions about sentiment persistence. While forward-filling helped, it may introduce bias, which we plan to address in future iterations.

- **Model Differences**: FinBERT and RoBERTa produced slightly different sentiment distributions, likely due to their training data and tokenization approaches. Reconciling these differences for a unified analysis remains a work in progress.








<!--
## How to Include a Link and Python

We chose [Investing.com](http://www.investing.com) to get the whole
year data of XRP and recalculated the return and 30 days volatility.

The code we use is as follows:
```python
import nltk
import pandas as pd
myvar = 8
DF = pd.read_csv('XRP-data.csv')
```


## How to Include a Quote

As a famous hedge fund manager once said:
>Fed watching is a great tool to make money. I have been making all my
>gazillions using this technique.



## How to Include an Image

Fed Chair Powell is working hard:

![Picture showing Powell]({static}/images/group-Fintech-Disruption_Powell.jpeg)
-->