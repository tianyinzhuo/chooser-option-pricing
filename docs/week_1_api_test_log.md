# Week 1 API Test Log

## Test Date

2026-08-03

## Yahoo Finance

- Tool: `yfinance` and Yahoo Finance chart endpoint
- Instruments tested: JPM and VIX
- Result: Unavailable in the local network environment
- Issue: The Yahoo Finance chart endpoint returned `HTTP 000`.
- Decision: Not used in the current data pipeline.

## Alpha Vantage

- Endpoint tested: `TIME_SERIES_DAILY`
- Instrument: JPM
- Network status: Reachable after invalid proxy settings were removed.
- API result: The free account returned an API information/limit response.
- Decision: API key remains stored only in `.env`; this source is not used
  in the current data pipeline.

## FRED

- Series tested: `VIXCLS` and `DGS10`
- Result: Successful
- Output:
  - `vix_daily_2018_2024.csv`
  - `treasury_10y_daily_2018_2024.csv`

## Twelve Data

- Endpoint tested: `time_series`
- Instrument: JPM
- Result: Successful
- Output: `jpm_daily_2018_2024.csv`
- Coverage: 2018-01-02 to 2024-12-31, 1,761 trading-day observations.
- Decision: Used as the JPM daily OHLCV source for the project.