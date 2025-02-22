---
Title: First Blog Post (by Group "Non JaneStreet")
Date: 2025-02-23 23:59
Category: Reflective Report
Tags: Group Fintech Disruption
---

This is the first blog post. Non Jane Street

## Connect to the Server

The server is a clickhouse DBMS. Difference between clickhouse and traditional MySQL server:

"The main difference between ClickHouse and MySQL 
is that ClickHouse is a columnar database optimized 
for high-performance analytics and large-scale 
data processing, while MySQL is a row-based 
relational database designed for transactional 
processing and smaller-scale applications."

Use clickhouse_driver to connect to the server.

Installation: 
```python
pip install clickhouse-driver    # Native protocol (recommended for performance)
pip install clickhouse-connect   # HTTP/HTTPS protocol
```

We use codes below to connect to the server so that we can gain an accessment to our news data for afterwards steps. 
```python
client = Client(
    host=host_name,
    user=user_name,
    password=pswd,
    database=db_name,
    port=port,
)
```

Here is a snapshot of our news data:
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
