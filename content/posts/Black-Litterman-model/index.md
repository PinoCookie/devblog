+++
date = '2025-12-04T18:20:00+01:00'
draft = false
title = 'Black Litterman Model'
+++

{{< katex >}}


## Introduction
For my first project in the "Quant AI" space I started with one of the easier, but still somewhat challenging project of making a Black-Litterman model that would be able to show the best possible stock allocation based on the market data of the last few years.

## Laying the ground work
At first I started of the project getting back into the feeling of using python for data science purposes, since because of my previous study major I did not use python for this goal as much. Because of this I build the project step-by-step

### Methodology

#### Black-Litterman
The main part of this project for me was understanding the Black-Litterman model itself.

$$
E(R) = \left[ (\tau \Sigma)^{-1} + P^{T} \Omega^{-1} P \right]^{-1} 
       \left[ (\tau \Sigma)^{-1} \Pi + P^{T} \Omega^{-1} Q \right]
$$

Where:

- \( E(R) \): \(N \times 1\) vector of posterior expected returns (number of assets = \(N\))
- \( Q \): \(K \times 1\) vector of views
- \( P \): \(K \times N\) picking matrix mapping views to assets
- \( \Omega \): \(K \times K\) diagonal covariance (uncertainty) matrix of views
- \( \Pi \): \(N \times 1\) vector of prior (equilibrium) expected returns
- \( \Sigma \): \(N \times N\) covariance matrix of asset returns
- \( \tau \): scalar tuning constant (reflecting uncertainty in the prior)


This gives us a good starting point for the code.


#### Python
For this project we will be making use of the [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/index.html) module, this module will make the overal calculations a lot easier.
```python
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import EfficientFrontier, objective_functions
```

Then for the data I used the [yfinance](https://pypi.org/project/yfinance/) module, this module gives me easy access to all the market data based on tickers. As of tickers I choose:
```python
import yfinance as yf
```

As of tickers I choose
```python
tickers = ["MSFT", "AMZN", "NVDA", "LLY", "AVGO", "PLTR", "GE", "STX", "HWM", "UNP", "MA", "V", "ADP", "DE", "LMT"]
```

From this I retrieved the data and filtered the data to training data (up until this year) and test data (this year)
```python
data = yf.download(tickers + ['SPY'], period="3y")['Close']

current_year_mask = pd.to_datetime(data.index) > pd.to_datetime('2024-12-31')
training_data = data[~current_year_mask].loc[:, data.columns != 'SPY']
training_market_prices = data[~current_year_mask]['SPY']

current_year_data = data[current_year_mask].loc[:, data.columns != 'SPY']
current_year_spy = data[current_year_mask]['SPY']
```

Now we can start calculating, the first caluclation we will do is find the covariance matrix for the stocks that we choose.
```python
S = risk_models.CovarianceShrinkage(training_data).ledoit_wolf()
```
The variance matrix we will shrink using Ledoit and Wolf proposed method in ["A well-conditioned estimator for large-dimensional covariance matrices"](https://www.sciencedirect.com/science/article/pii/S0047259X03000964)

![Covariance matrix based on the stocks chosen](covariance.png)

After this we start calculating the market priors.
```python
delta = black_litterman.market_implied_risk_aversion(training_market_prices)

market_prior = black_litterman.market_implied_prior_returns(ticker_market_caps, delta, S)
```

Then the P Q and confidence levels
```python
ai_stocks = [0,1,2,4,5,7]

q = np.array([0.10]).reshape(-1,1)
p = np.array(
    [[1/6,1/6,1/6,-1/9,1/6,1/6,-1/9,1/6,-1/9,-1/9,-1/9,-1/9,-1/9,-1/9,-1/9]]
)

confidences = [0.4]
```
For this I used the current hype around AI stocks.

With this we were able to calculate the stock allocation proposed by the model
```python
ef = EfficientFrontier(post_returns_bl, bl.bl_cov())
ef.add_objective(objective_functions.L2_reg)
ef.max_sharpe()
weights = ef.clean_weights()
weights
pd.Series(weights).plot.pie(figsize=(10,10));
```
![A pie chart showing the stock allocation](stock_allocation.png)

After this doing some backtesting of the S&P 500 (SPY) and my own portifolio, I got these results
![A graph showing the movement of the stocks](first_backtesting.png)
With a sharpe ratio of ...
### Results
From this I can conclude that the first phase of the project was definetly a success, the model was able to outperform the S&P 500 with quite the amount. This does not mean that everyone should now invest in the portfolio I calculated, because my personal views were heavily based on what I know this year brought. But it does show the impact of being able to add your own views into the mix when determining how to allocate your portfolio.

## Everything needs AI
Now that the manual work has been done and I have an understanding of the subject, I wanted to add AI into the mix and let them compete against one another to see who would be able to best predict the market.

### Methodology
This part of the project took by far the longest, this was because I needed to find out how to get AI's to behave in the exact way that I want and return me the information I was looking for. At the current stage of AI this is still quite the ask to make. So for this I created a python module to help me out with some of the parts. In this model I build 3 tools for the AI's to use while trying to come up with an answer. These tools consisted of a search engine, this engine was a locally hosted searchxng container which they were able to access through a query. Then I gave them access to the previously mentioned yfinance module, through this they were able to get market information on by them requested stocks, and lastely a P and Q validation tool. This tool was added later into the development, because I kept bumping my head into the problem that the AI's were struggling with formulating these variables correctly.

### Results
This time I will not bore you with all the little steps but immediatly show you the results of the great AI Stock trading portfolio allocation competition....

First of their allocations.

![alt text](AI_allocations.png)

And now the backtesting results.

![alt text](AI_backtesting.png)

With sharpe ratios of:

```cli
-------------------------------------------------------
Model                      Return    Sharpe Ratio
-------------------------------------------------------
GPT-5 mini                 37.88%            1.14
Qwen3                      25.32%            0.82
Grok 4.1 fast              33.45%            0.96
-------------------------------------------------------
S&P 500                    18.09%            0.86
```

So in the end not that promissing...

### Discussions