"""Build leakage-safe JPM news-sentiment features for Week 7."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("CHOOSER_PROJECT_ROOT", DEFAULT_ROOT)).resolve()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "config" / "week_7"

ARTICLES_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "news_sentiment"
    / "jpm_news_articles_2018_2024.csv"
)
TRADING_CALENDAR_PATH = PROCESSED_DIR / "aligned_clean_data.csv"
BASE_FEATURES_PATH = PROCESSED_DIR / "feature_dataset_2018_2024.csv"
OUTPUT_PATH = Path(
    os.environ.get(
        "WEEK7_FEATURE_OUTPUT",
        str(PROCESSED_DIR / "week_7_feature_dataset_with_sentiment.csv"),
    )
).resolve()
METADATA_PATH = Path(
    os.environ.get(
        "WEEK7_SENTIMENT_METADATA",
        str(CONFIG_DIR / "sentiment_feature_config.json"),
    )
).resolve()

SENTIMENT_FEATURES = [
    "news_available_lag1",
    "news_article_count_lag1",
    "news_sentiment_mean_lag1",
    "news_sentiment_weighted_lag1",
    "news_positive_share_lag1",
    "news_negative_share_lag1",
    "news_neutral_share_lag1",
    "news_sentiment_ma_5",
    "news_sentiment_ma_20",
    "news_article_count_ma_5",
]


def load_inputs() -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame]:
    articles = pd.read_csv(ARTICLES_PATH)
    calendar = pd.read_csv(TRADING_CALENDAR_PATH, usecols=["date"])
    features = pd.read_csv(BASE_FEATURES_PATH, parse_dates=["date"])

    calendar_dates = pd.DatetimeIndex(
        pd.to_datetime(calendar["date"], errors="raise").drop_duplicates().sort_values()
    )
    if calendar_dates.empty or not calendar_dates.is_monotonic_increasing:
        raise ValueError("The JPM trading calendar is empty or unsorted.")
    return articles, calendar_dates, features


def map_articles_to_next_trading_day(
    articles: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, int]:
    """Map each UTC news date strictly to the next JPM trading date."""
    result = articles.copy()
    result["timestamp_utc"] = pd.to_datetime(
        result["time_published"],
        format="%Y%m%dT%H%M%S",
        utc=True,
        errors="coerce",
    )
    result["publication_date_utc"] = result["timestamp_utc"].dt.tz_localize(None).dt.normalize()
    result["jpm_sentiment_score"] = pd.to_numeric(
        result["jpm_sentiment_score"], errors="coerce"
    )
    result["jpm_relevance_score"] = pd.to_numeric(
        result["jpm_relevance_score"], errors="coerce"
    )
    result = result.dropna(
        subset=["publication_date_utc", "jpm_sentiment_score", "jpm_relevance_score"]
    )

    positions = trading_dates.searchsorted(
        pd.DatetimeIndex(result["publication_date_utc"]), side="right"
    )
    within_calendar = positions < len(trading_dates)
    excluded_after_calendar = int((~within_calendar).sum())
    result = result.loc[within_calendar].copy()
    positions = positions[within_calendar]
    result["feature_date"] = trading_dates.take(positions).to_numpy()
    return result, excluded_after_calendar


def aggregate_to_trading_dates(mapped: pd.DataFrame) -> pd.DataFrame:
    label = mapped["jpm_sentiment_label"].fillna("").str.lower()
    mapped = mapped.copy()
    mapped["positive"] = label.str.contains("bullish", regex=False).astype(float)
    mapped["negative"] = label.str.contains("bearish", regex=False).astype(float)
    mapped["neutral"] = label.str.contains("neutral", regex=False).astype(float)
    mapped["weighted_score"] = (
        mapped["jpm_sentiment_score"] * mapped["jpm_relevance_score"]
    )

    grouped = (
        mapped.groupby("feature_date", as_index=False)
        .agg(
            news_article_count_lag1=("jpm_sentiment_score", "size"),
            news_sentiment_mean_lag1=("jpm_sentiment_score", "mean"),
            weighted_score_sum=("weighted_score", "sum"),
            relevance_sum=("jpm_relevance_score", "sum"),
            news_positive_share_lag1=("positive", "mean"),
            news_negative_share_lag1=("negative", "mean"),
            news_neutral_share_lag1=("neutral", "mean"),
        )
        .rename(columns={"feature_date": "date"})
    )
    grouped["news_sentiment_weighted_lag1"] = np.divide(
        grouped["weighted_score_sum"],
        grouped["relevance_sum"],
        out=np.zeros(len(grouped), dtype=float),
        where=grouped["relevance_sum"].to_numpy() != 0,
    )
    return grouped.drop(columns=["weighted_score_sum", "relevance_sum"])


def build_daily_sentiment_panel(
    trading_dates: pd.DatetimeIndex,
    grouped: pd.DataFrame,
) -> pd.DataFrame:
    panel = pd.DataFrame({"date": trading_dates}).merge(grouped, on="date", how="left")
    fill_columns = [
        "news_article_count_lag1",
        "news_sentiment_mean_lag1",
        "news_sentiment_weighted_lag1",
        "news_positive_share_lag1",
        "news_negative_share_lag1",
        "news_neutral_share_lag1",
    ]
    panel[fill_columns] = panel[fill_columns].fillna(0.0)
    panel["news_article_count_lag1"] = panel["news_article_count_lag1"].astype(int)
    panel["news_available_lag1"] = (panel["news_article_count_lag1"] > 0).astype(int)
    panel["news_sentiment_ma_5"] = panel["news_sentiment_weighted_lag1"].rolling(
        5, min_periods=1
    ).mean()
    panel["news_sentiment_ma_20"] = panel["news_sentiment_weighted_lag1"].rolling(
        20, min_periods=1
    ).mean()
    panel["news_article_count_ma_5"] = panel["news_article_count_lag1"].rolling(
        5, min_periods=1
    ).mean()
    return panel[["date", *SENTIMENT_FEATURES]]


def main() -> None:
    articles, trading_dates, base_features = load_inputs()
    mapped, excluded_after_calendar = map_articles_to_next_trading_day(
        articles, trading_dates
    )
    grouped = aggregate_to_trading_dates(mapped)
    sentiment_panel = build_daily_sentiment_panel(trading_dates, grouped)
    result = base_features.merge(sentiment_panel, on="date", how="left", validate="one_to_one")

    if result[SENTIMENT_FEATURES].isna().any().any():
        missing = result[SENTIMENT_FEATURES].isna().sum()
        raise RuntimeError(f"Missing sentiment features remain:\n{missing[missing > 0]}")
    if result["date"].duplicated().any():
        raise RuntimeError("Duplicate market dates were created.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    metadata = {
        "experiment_name": "week_7_leakage_safe_news_sentiment_features",
        "source": "Alpha Vantage NEWS_SENTIMENT filtered for JPM",
        "source_articles": int(len(articles)),
        "mapped_articles": int(len(mapped)),
        "excluded_after_last_trading_date": excluded_after_calendar,
        "output_rows": int(len(result)),
        "output_date_range": [
            result["date"].min().date().isoformat(),
            result["date"].max().date().isoformat(),
        ],
        "trading_days_with_prior_news": int(result["news_available_lag1"].sum()),
        "sentiment_features": SENTIMENT_FEATURES,
        "anti_leakage_rule": (
            "Each article is assigned to the first JPM trading day strictly after "
            "its UTC publication date. Weekend and holiday news rolls forward."
        ),
        "missing_news_rule": (
            "No-news trading days receive article_count=0, sentiment=0, and "
            "news_available_lag1=0."
        ),
        "disclosures": [
            "Alpha Vantage sentiment is an external vendor score.",
            "UTC calendar-day mapping is conservative and avoids same-day after-close leakage.",
            "A zero score on no-news days means no observed JPM article, not proven neutral sentiment.",
        ],
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Week 7 sentiment-feature construction completed.")
    print(f"Source articles: {len(articles)}")
    print(f"Mapped articles: {len(mapped)}")
    print(f"Excluded after last trading date: {excluded_after_calendar}")
    print(f"Output rows: {len(result)}")
    print(
        "Output date range: "
        f"{result['date'].min().date()} to {result['date'].max().date()}"
    )
    print(f"Trading days with prior news: {int(result['news_available_lag1'].sum())}")
    print(f"Sentiment feature count: {len(SENTIMENT_FEATURES)}")
    print("Total missing sentiment values: 0")
    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print("Anti-leakage rule: every article is mapped strictly to a later trading day.")


if __name__ == "__main__":
    main()
