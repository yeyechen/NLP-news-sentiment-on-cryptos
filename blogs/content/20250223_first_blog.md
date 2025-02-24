---
Title: First Blog Post (by Group "Non JaneStreet")
Date: 2025-02-23 23:59
Category: Reflective Report
Tags: Group Non Jane Street
---

## Introduction

In this blog post, we explore the use of FinBERT, a transformer model for financial sentiment analysis, to analyze news data. We will walk through how to connect to a ClickHouse database to retrieve financial news, use FinBERT to analyze sentiment, and explain how we process the results for further analysis. By the end, you'll understand how FinBERT can help derive insights from news articles and potentially assist in financial decision-making processes.

## Setting Up the ClickHouse Connection

To begin our analysis, we first need to connect to a ClickHouse database, which contains the financial news articles. ClickHouse is a columnar database designed for high-performance analytics, making it suitable for our large-scale data processing needs.

### Key Difference Between ClickHouse and MySQL

- **ClickHouse** is optimized for fast, real-time data processing and analytics. It stores data in columns rather than rows, allowing for quick aggregation and retrieval of large datasets.
- **MySQL**, on the other hand, is a traditional row-based relational database primarily used for transactional systems and smaller datasets.

For our project, we use the `clickhouse_driver` to connect to the ClickHouse server and retrieve news data.

### Installing Required Packages

To get started, install the necessary libraries:
```python
pip install clickhouse-driver    # Native protocol (recommended for performance)
pip install clickhouse-connect   # HTTP/HTTPS protocol
```
### Code to Connect to ClickHouse
```python
from clickhouse_driver import Client

client = Client(
    host=host_name,
    user=user_name,
    password=pswd,
    database=db_name,
    port=port,
)
```
Once connected, we can query the server for our news data. Here's a snapshot of one of the news records we retrieved:
```python
[('2018-05-02T12:14:50.841934+00:00',
  'Yamana Gold Inc. (NYSE:AUY) shares are down more than -8.65% this year and recently decreased -0.70% or -$0.02 to settle at $2.85. SM Energy Company (NYSE:SM), on the other hand, is up 6.61% year to date as of 05/01/2018. It currently trades at $23.54 and has returned 3.11% during the past week. Yamana Gold Inc.…',
  10471811.0,
  '2018-05-02T11:06:33+00:00',
  'stocknewsgazette.com',
  'Energy/Materials/Stock',
  'sm',
  'auy/sm',
  'Yamana Gold Inc. (AUY) vs. SM Energy Company (SM): Which is the Better Investment?',
  'stocknewsgazette.com',
  'https://stocknewsgazette.com/2018/05/02/yamana-gold-inc-auy-vs-sm-energy-company-sm-which-is-the-better-investment/')]
```

## Testing for FinBERT

Now that we have the data, we turn our attention to sentiment analysis using **FinBERT**, a BERT-based model fine-tuned for financial data. FinBERT is ideal for extracting sentiment from news articles, which can be positive, negative, or neutral.

### Setting Up the Environment

To run FinBERT, ensure you have the following libraries installed:

```python
pip install torch==2.2.2
pip install pandas==2.2.1
pip install numpy==1.26.4
pip install scipy==1.13.0
pip install huggingface-hub==0.29.1
```
### Importing Libraries and Checking PyTorch Version

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import scipy
import torch

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')  # Should be True on servers
print(f'CUDA version: {torch.version.cuda}')  # Should show 12.1 on servers
```

Make sure that your server supports CUDA, as this will speed up the model inference process.

### Design Choice
We now have two choices when it comes to using Hugging face open-source models:

- **Option 1**: use `pipeline()`

```python
classifier = pipeline("text-classification", model="ProsusAI/finbert")
```

Using `pipeline()` is more convenient and easier to use, it is generally for quick prototyping and simple tasks.

- **Option 2**: use direct model approach
We need to create a tokenizer and a model like this

```python
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
```
We choose this approach because we want to have more control over the model, for performance optimisation and deployment. 

### Deploying the FinBERT Model

We create a upstream function to calculate sentiment scores for a specific column (text column) in a given DataFrame. It returns the sentiment scores for positive, negative, and neutral, along with the predominant sentiment (positive, negative, or neutral). This function will be like the following:

```python
def finbert_sentiment_fill_df(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    tokenizer_finbert = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model_finbert = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

    copy_df = df.copy()
    copy_df[['finbert_pos', 'finbert_neg', 'finbert_neu',
             'finbert_stmt_score', 'finbert_stmt_label']] = (
        copy_df[text_col].apply(lambda x: calc_sentiment(x, 
        tokenizer_finbert,model_finbert, model_name='finbert')).apply(pd.Series)
    )
    return copy_df
```
where the downstream function will look like this:

```python
def calc_sentiment(text: str, 
                  tokenizer: AutoTokenizer, 
                  model: AutoModelForSequenceClassification, 
                  model_name: str) -> tuple:
    with torch.no_grad():
        inputs = tokenizer(
            text, return_tensors='pt', padding=True, truncation=True, max_length=1024
        )
        outputs = model(**inputs)

        # convert logits to probabilities
        values = scipy.special.softmax(outputs.logits.numpy().squeeze())
        labels = [*model.config.id2label.values()]
        result = (
            values[0], # positive
            values[1], # negative
            values[2], # neutral
            values[0] - values[1], # sentiment score
            labels[np.argmax(values)], # sentiment label (max probability label)
        )

        return result
```
The above funtion is specifically for FinBERT, different pre-trained classification models will have different labels, for example, the **DeBERTa-v3** model will only have positive and negative labels. We need to potentially modify the downstream `calc_sentiment()` to further incoperate different models. We will have `deberta_sentiment_fill_df()` and `deepseek_sentiment_fill_df()` in our further development.

We define a function to run the sentiment analysis on each news article:
```python
def finbert_sentiment(text: str) -> tuple[float, float, float, str]:
    with torch.no_grad():
        inputs = tokenizer(
            text, return_tensors='pt', padding=True, truncation=True, max_length=512
        )
        outputs = model(**inputs)
        logits = outputs.logits
        scores = {
            k: v
            for k, v in zip(
                model.config.id2label.values(),
                scipy.special.softmax(logits.numpy().squeeze()),
            )
        }
        return (
            scores['positive'],
            scores['negative'],
            scores['neutral'],
            max(scores, key=scores.get),
        )
```

Here is our sample output of the three models:

## Table 1: Sentiment Analysis Results

| crawlDate            | description                                                                 | sst2_label |
|----------------------|-----------------------------------------------------------------------------|-----------------|
| 2022-09-08 03:59:27 | BALTIMORE (AP) — Alek Manoah retiró a 22 de su...                          | negative        |
| 2022-09-08 03:59:24 | CHICAGO (AP) — El dominicano Arístides Aquino ...                          | positive        |
| 2022-09-08 03:59:22 | Prime Video has released a statement.                                      | negative        |
| 2022-09-08 03:59:21 | KANSAS CITY, Mo. (AP) — Salvador Perez’s sacri...                          | positive        |
| 2022-09-08 03:58:26 | North West Earnings Miss, Revenue Beats In Q2                              | negative        |

---

## Table 2: FinBERT Analysis

| crawlDate            | finbert_pos | finbert_neg | finbert_neu | finbert_score | finbert_label |
|----------------------|-------------|-------------|-------------|--------------------|--------------------|
| 2022-09-08 03:59:27 | 0.410662    | 0.019876    | 0.569462    | 0.390786           | neutral            |
| 2022-09-08 03:59:24 | 0.130872    | 0.018290    | 0.850838    | 0.112582           | neutral            |
| 2022-09-08 03:59:22 | 0.020164    | 0.054695    | 0.925141    | -0.034531          | neutral            |
| 2022-09-08 03:59:21 | 0.617857    | 0.034792    | 0.347351    | 0.583064           | positive           |
| 2022-09-08 03:58:26 | 0.525914    | 0.450251    | 0.023835    | 0.075663           | positive           |

---

## Table 3: DeBERTa Analysis

| crawlDate            | deberta_pos | deberta_neg | deberta_score | deberta_label |
|-----------------|-------------|-------------|---------------------------|--------------------|
| 2022-09-08 03:59:27 | 0.517502    | 0.482498    | 0.035004           | positive           |
| 2022-09-08 03:59:24 | 0.515786    | 0.484214    | 0.031571           | positive           |
| 2022-09-08 03:59:22 | 0.520406    | 0.479594    | 0.040811           | positive           |
| 2022-09-08 03:59:21 | 0.518034    | 0.481966    | 0.036069           | positive           |
| 2022-09-08 03:58:26 | 0.515076    | 0.484924    | 0.030152           | positive           |

This code will add the sentiment analysis results to the DataFrame, allowing us to assess the sentiment of each news article. The `finbert_score` column gives us a quick indication of the overall sentiment, with a positive score indicating a positive sentiment and a negative score indicating a negative sentiment.

## Model Comparison

| Model    | Pos/Neg/Neu | Score Range | Label Logic                                   | Example Performance (Row 4)                       |
|----------|-------------|-------------|-----------------------------------------------|---------------------------------------------------|
| **FinBERT** | ✓✓✓        | -1 to +1    | Neutral if max(neu) > pos+neg                 | Correctly labels earnings news (positive)        |
| **SST-2**   | ✓✓          | 0-1         | Simple argmax(pos/neg)                        | Mislabels earnings news (negative)               |
| **DeBERTa** | ✓✓          | -1 to +1    | Always positive label in your data            | Overly optimistic (labels all positive)          |

---

### Key Observations

#### **FinBERT (Recommended for Financial Texts)**

**Strengths:**

- **Neutral class detection**: 0.925 neutral probability in Row 2 (TV news)

- **Financial context awareness**: Correct positive label for earnings news (Row 4)
- **Conservative scoring**: 0.583 score for clear positive case (Row 3)
---

#### **SST-2 (Not Recommended)**

**Issues:**

- **Binary classification fails on financial nuance**

- **Contradicts others**: Labels earnings beat (Row 4) as negative

- **No neutral class**: Forces positive/negative even for factual reports

---

#### **DeBERTa (Use with Caution)**

**Patterns:**

- **Always positive labels in your sample**: Positive in all 5 rows

- **Tight score range**: All scores between 0.03-0.04 difference

- **Potential overconfidence**: Labels TV controversy (Row 2) as positive

## Conclusion of Our First Blog

In our financial sentiment analysis project, we rigorously evaluated three specialized models: BERT Base SST-2, FinBERT, and DeBERTa. Based on benchmark results from AI4Finance's FinGPT repository, FinBERT demonstrated superior performance on the Financial Phrase Bank Dataset (FPB) with a score of 0.880 compared to models like GPT4(0.833) and BloombergGPT(0.511). This aligns with our requirement for precise detection of financial-specific semantics like "earnings surprise" and "regulatory overhang." While DeBERTa showed promise in general language contexts, FinBERT's domain-specific training in financial lexicon made it our benchmark choice.

A key benefit of FinBERT is the structure of its three-class output. In contrast, binary classifiers like SST-2 (0.636 TFNS score) applied forced artificial positive/negative labels to factual statements like “The Fed left the rates at 5.25%.” FinBERT’s neutral classification, with a TFNS score of 0.538, lets us separate factual financial reporting from actionable market sentiment—a critical requirement for our stakeholders.

## Further Work
To further enhance our sentiment analysis capabilities, we plan a three-phase evaluation of DeepSeek-v3 and Ollama 3.1 while maintaining our FinBERT production pipeline. We will run parallel inferences on a 1,000-article sample from our news dataset, comparing sentiment labels from each model against our FinBERT baseline and human-annotated ground truth. Models achieving at least 85% accuracy relative to human labels will be considered for integration into a hybrid fallback system, where low-confidence FinBERT predictions (scores between 0.3-0.7) trigger additional analysis from larger models.

Furthermore, we will assign sentiment scores to news articles that reference specific stock tickers and merge these scores with our Bitcoin hourly dataset. By averaging sentiment scores within each hourly window, we can analyze potential correlations through regression modeling, providing deeper insights into sentiment-driven market movements.






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