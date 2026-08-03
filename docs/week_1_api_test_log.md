\# Week 1 API Test Log



\## Test Date



2026-08-03



\## Yahoo Finance



\- Tool: `yfinance`

\- Instruments tested: JPM and ^VIX

\- Result: Failed

\- Issue: Yahoo Finance data endpoints timed out in the local network environment.

\- Decision: Not used for the project data pipeline.



\## Alpha Vantage



\- Endpoint tested: `TIME\_SERIES\_DAILY`

\- Instrument: JPM

\- Result: Network unavailable

\- Issue: Local DNS could not resolve `www.alphavantage.co`.

\- Decision: API key configuration is retained in `.env`; the source is not used

&#x20; in the current pipeline.



\## FRED



\- Series tested: `VIXCLS`, `DGS10`

\- Result: Successful

\- Output:

&#x20; - `vix\_daily\_2018\_2024.csv`

&#x20; - `treasury\_10y\_daily\_2018\_2024.csv`



\## Twelve Data



\- Endpoint tested: `time\_series`

\- Instrument: JPM

\- Result: Successful

\- Output: `jpm\_daily\_2018\_2024.csv`

\- Coverage: 2018-01-02 to 2024-12-31, 1,761 trading-day observations.

\- Decision: Used as the JPM daily OHLCV source for the project.

