\# Data Specification — Chooser Option Pricing Model



\## 1. Purpose



This document defines the initial data requirements for building and validating

a chooser option pricing model. The first data collection stage covers daily

market, volatility, and interest-rate data from 1 January 2018 to

31 December 2024.



\## 2. Instruments and Data Sources



| Dataset | Symbol / Series | Source | Frequency | Purpose |

|---|---|---|---|---|

| JPMorgan Chase stock price | JPM | Twelve Data | Daily | Underlying asset price and return calculation |

| CBOE Volatility Index | VIXCLS | FRED | Daily | Market-implied volatility and market-risk proxy |

| 10-Year Treasury Constant Maturity Rate | DGS10 | FRED | Daily | Risk-free-rate proxy for Black-Scholes pricing |



\## 3. Required Raw Fields



\### JPM Daily OHLCV



\- Date

\- Open

\- High

\- Low

\- Close

\- Volume



\### VIX



\- Observation date

\- VIX closing level



\### 10-Year Treasury Rate



\- Observation date

\- Daily 10-year Treasury constant maturity rate



\## 4. Date Range and Frequency



\- Start date: 2018-01-01

\- End date: 2024-12-31

\- Frequency: Daily

\- Trading calendar: U.S. market trading days



\## 5. Data Storage



Raw files are stored in `data/raw/`.



\- `jpm\_daily\_2018\_2024.csv`

\- `vix\_daily\_2018\_2024.csv`

\- `treasury\_10y\_daily\_2018\_2024.csv`



\## 6. Data Quality Notes



\- JPM contains 1,761 trading-day observations with no missing values.

\- VIX and Treasury data include observations for calendar days; missing values

&#x20; correspond mainly to weekends and U.S. market holidays.

\- Missing values will be handled during Week 2 preprocessing after the series

&#x20; are aligned to a common trading calendar.

\- API keys are stored only in `.env` and must never be uploaded to GitHub.



\## 7. Planned Week 2 Features



\- Daily and log returns

\- Rolling realized volatility

\- VIX-JPM correlation

\- Interest-rate momentum

\- Dividend-related variables

\- Sentiment score

\- Time-to-expiry and option contract parameters

