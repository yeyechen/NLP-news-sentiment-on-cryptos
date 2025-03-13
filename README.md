# U.S. Market Sentiment Effects on Cryptocurrency Returns

## Overview
This project investigates the predictive power of U.S. market sentiment on cryptocurrency returns using advanced Natural Language Processing (NLP) techniques. By analyzing over 43 million financial news articles, the study employs state-of-the-art transformer models (FinBERT, RoBERTa, DeBERTa) to decode sentiment signals and develop a dynamic trading strategy. Key findings reveal that domain-specific models like FinBERT outperform general-purpose models, and negative U.S. market sentiment often precedes positive cryptocurrency returns.

## Key Features
- **Multi-Model Sentiment Analysis**: Utilizes FinBERT, RoBERTa, and DeBERTa for robust sentiment extraction.
- **Market-Cap Weighted Aggregation**: Novel methodology to aggregate sentiment scores, accounting for news coverage disparities and sector-specific effects.
- **Trading Strategy**: Combines sentiment signals with price momentum to dynamically adjust positions, outperforming buy-and-hold strategies for BTC and ETH.
- **Validation Against CNN Fear & Greed Index**: Granger causality tests and VAR models confirm alignment with established market sentiment indicators.

## Methodology
### Data Collection
- **Dataset**: 43 million news articles from Tiingo, filtered for U.S. stocks.
- **Preprocessing**: Timezone standardization, duplicate removal, and text cleaning.

### Sentiment Analysis Pipeline
1. **Model Implementation**: PyTorch and Hugging Face Transformers for inference.
2. **Score Aggregation**:
   - Market-cap weighting using CRSP data.
   - Sector-specific analysis (GICS classification).
   - Forward-filling and penalization for sparse data.

### Trading Strategy
- **Signal Generation**: Combines 20-day moving averages of sector sentiment (Industrials, Utilities, Real Estate) with price momentum filters.
- **Position Management**: Automatic liquidation after 5 periods without refresh.

## Results
- **Model Performance**: FinBERT shows superior sector-specific correlations (e.g., -0.13 in Industrials) compared to RoBERTa and DeBERTa.
- **Strategy Returns**: Outperforms passive holding of BTC/ETH, with reduced volatility and drawdowns during bear markets (2018, 2022).
- **Validation**: Granger causality tests confirm predictive power of sentiment scores (p < 0.05 for lags 1–5).