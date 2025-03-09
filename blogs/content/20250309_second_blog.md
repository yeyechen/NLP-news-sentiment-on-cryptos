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
df['date'] = df['date'] = [x[:10] for x in df['publishedDate']]
df = df[df['date'] >= '2017-08-17']
```

Additionally, we used ticker count data (`sNv.pkl`) to understand the number of unique stocks mentioned per day and per sector, which helped us normalize sentiment scores later. This step ensured we accounted for varying news coverage across different stocks and sectors.

### Sentiment Clustering Methodology

To aggregate sentiment scores, we developed a custom `sentiment_cluster` function that addressed two key challenges:

- **Avoiding Double Counting:** Some stocks had multiple news articles on the same day, which could skew sentiment scores. We averaged sentiment scores for each stock on a given day to mitigate this.

- **Handling Missing News:** For stocks without news on a particular day, we assumed sentiment remained the same as the previous period, using forward-filling to propagate scores.

Here’s a simplified version of our clustering logic:

```python
def sentiment_cluster(df_, stock_counts, analysis_on='finbert', freq='d', impact=3):
    df_ = df_.copy()
    freq_col = 'date'
    pos_col = analysis_on + '_pos'
    neg_col = analysis_on + '_neg'
    neu_col = analysis_on + '_neu'
    
    # Average sentiment scores per ticker and date
    temp_senti_s = df_.groupby(['ticker', freq_col]).apply(
        lambda x: x[[pos_col, neg_col, neu_col]].mean(axis=0)
    )
    temp_senti_freq = temp_senti_s.groupby(level=1).mean()
    temp_count_freq = temp_senti_s.groupby(level=1).apply(lambda x: x.shape[0])
    temp_count_freq.name = 'inner_count'
    
    # Merge with stock counts and forward-fill missing values
    stock_counts.name = 'count'
    temp_merge = pd.concat([stock_counts, temp_count_freq, temp_senti_freq], axis=1).sort_index().ffill()
    
    # Weighted averaging with impact factor
    cluster_dict = {pos_col: [], neg_col: [], neu_col: []}
    last_senti = {pos_col: temp_merge.head(1)[pos_col].values[0],
                  neg_col: temp_merge.head(1)[neg_col].values[0],
                  neu_col: temp_merge.head(1)[neu_col].values[0]}
    
    for row in temp_merge.iterrows():
        for s in [pos_col, neg_col, neu_col]:
            temp_put = (row[1][s] * impact * row[1]['inner_count'] + 
                        last_senti[s] * (row[1]['count'] - row[1]['inner_count'])) / \
                       (impact * row[1]['inner_count'] + row[1]['count'] - row[1]['inner_count'])
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
        temp_d = sentiment_cluster(group, tc, analysis_on='roberta', freq='d')
        sec_dict[sector] = temp_d
    return sec_dict
```

The output was a dictionary of DataFrames, each containing daily sentiment scores for a specific sector. We saved these results as pickled files (`roberta_sector_sentiment.pkl` and `finbert_sentiment_by_sector_dict.pkl`) for future analysis.

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