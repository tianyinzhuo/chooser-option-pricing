"""Week 6 time-aware cross-validation utilities."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def make_volatility_cv(
    n_rows: int,
    n_splits: int = 5,
    gap_rows: int = 20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create chronological CV folds for future-volatility prediction.

    The gap prevents the future 20-day target window in the training set
    from overlapping the validation period.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be greater than zero.")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if gap_rows < 0:
        raise ValueError("gap_rows cannot be negative.")

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap_rows,
    )

    row_indices = np.arange(n_rows)
    folds = list(splitter.split(row_indices))

    for train_indices, validation_indices in folds:
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise ValueError("Every fold must contain training and validation rows.")

        if train_indices.max() >= validation_indices.min():
            raise ValueError("Training rows must occur before validation rows.")

        actual_gap = validation_indices.min() - train_indices.max() - 1

        if actual_gap < gap_rows:
            raise ValueError("The required time-series gap was not preserved.")

    return folds


def make_grouped_date_cv(
    dates: Sequence,
    n_splits: int = 4,
    gap_groups: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create expanding-window folds while keeping each date together.

    This is required because the synthetic pricing dataset contains
    multiple option contracts for each market date.
    """
    date_series = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)

    if date_series.empty:
        raise ValueError("dates cannot be empty.")

    if date_series.isna().any():
        raise ValueError("dates cannot contain missing values.")

    if not date_series.is_monotonic_increasing:
        raise ValueError("dates must be sorted chronologically.")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if gap_groups < 0:
        raise ValueError("gap_groups cannot be negative.")

    unique_dates = pd.Index(date_series.drop_duplicates())

    if len(unique_dates) <= n_splits:
        raise ValueError(
            "The number of unique dates must be greater than n_splits."
        )

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap_groups,
    )

    folds = []

    for train_date_indices, validation_date_indices in splitter.split(unique_dates):
        train_dates = unique_dates[train_date_indices]
        validation_dates = unique_dates[validation_date_indices]

        train_rows = np.flatnonzero(
            date_series.isin(train_dates).to_numpy()
        )
        validation_rows = np.flatnonzero(
            date_series.isin(validation_dates).to_numpy()
        )

        if len(train_rows) == 0 or len(validation_rows) == 0:
            raise ValueError("Every fold must contain training and validation rows.")

        train_date_set = set(date_series.iloc[train_rows])
        validation_date_set = set(date_series.iloc[validation_rows])

        if train_date_set & validation_date_set:
            raise ValueError(
                "The same market date appeared in both training and validation."
            )

        if date_series.iloc[train_rows].max() >= date_series.iloc[validation_rows].min():
            raise ValueError(
                "Training dates must occur strictly before validation dates."
            )

        folds.append((train_rows, validation_rows))

    return folds