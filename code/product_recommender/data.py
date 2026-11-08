"""Data loading and matrix-building utilities for the sample recommender."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix


EVENT_WEIGHTS = {
    "view": 1.0,
    "click": 2.0,
    "add_to_cart": 4.0,
    "purchase": 8.0,
    "review": 10.0,
}


@dataclass(frozen=True)
class MatrixMappings:
    """Keeps the original user and item IDs linked to matrix row/column indexes."""

    user_to_index: dict[str, int]
    item_to_index: dict[str, int]
    index_to_user: dict[int, str]
    index_to_item: dict[int, str]


def load_sample_data(base_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the included interaction and catalog CSV files."""

    base_path = Path(base_dir)
    interactions = pd.read_csv(base_path / "customer_product_interactions.csv")
    catalog = pd.read_csv(base_path / "product_catalog.csv")
    return interactions, catalog


def prepare_interaction_strengths(interactions: pd.DataFrame) -> pd.DataFrame:
    """Clean interaction rows and create a numeric implicit-feedback strength."""

    required_columns = {"user_id", "item_id", "event_type"}
    missing = required_columns.difference(interactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    prepared = interactions.copy()
    prepared["user_id"] = prepared["user_id"].astype(str)
    prepared["item_id"] = prepared["item_id"].astype(str)
    prepared["event_type"] = prepared["event_type"].str.lower().str.strip()

    if "event_strength" not in prepared.columns:
        prepared["event_strength"] = prepared["event_type"].map(EVENT_WEIGHTS)
    else:
        missing_strength = prepared["event_strength"].isna()
        prepared.loc[missing_strength, "event_strength"] = prepared.loc[
            missing_strength, "event_type"
        ].map(EVENT_WEIGHTS)

    if prepared["event_strength"].isna().any():
        unknown_events = prepared.loc[prepared["event_strength"].isna(), "event_type"].unique()
        raise ValueError(f"Unknown event_type values: {sorted(unknown_events)}")

    if "quantity" not in prepared.columns:
        prepared["quantity"] = 1
    if "interaction_datetime" not in prepared.columns:
        prepared["interaction_datetime"] = ""

    prepared["quantity"] = prepared["quantity"].fillna(1).clip(lower=1)
    prepared["interaction_strength"] = prepared["event_strength"].astype(float) * np.sqrt(
        prepared["quantity"].astype(float)
    )
    return prepared


def build_user_item_matrix(
    prepared_interactions: pd.DataFrame,
) -> tuple[csr_matrix, pd.DataFrame, MatrixMappings]:
    """Aggregate events into one sparse user-item matrix."""

    aggregated = (
        prepared_interactions.groupby(["user_id", "item_id"], as_index=False)
        .agg(
            interaction_strength=("interaction_strength", "sum"),
            event_count=("event_type", "size"),
            last_event_time=("interaction_datetime", "max"),
        )
        .sort_values(["user_id", "item_id"])
        .reset_index(drop=True)
    )

    # Log scaling keeps very active customers or repeat purchases from dominating.
    aggregated["matrix_value"] = np.log1p(aggregated["interaction_strength"])

    user_ids = sorted(aggregated["user_id"].unique())
    item_ids = sorted(aggregated["item_id"].unique())
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    rows = aggregated["user_id"].map(user_to_index).to_numpy()
    cols = aggregated["item_id"].map(item_to_index).to_numpy()
    values = aggregated["matrix_value"].to_numpy(dtype=np.float32)

    matrix = coo_matrix(
        (values, (rows, cols)), shape=(len(user_ids), len(item_ids)), dtype=np.float32
    ).tocsr()

    mappings = MatrixMappings(
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        index_to_user={idx: user_id for user_id, idx in user_to_index.items()},
        index_to_item={idx: item_id for item_id, idx in item_to_index.items()},
    )
    return matrix, aggregated, mappings


def train_test_split_by_user(
    user_item_matrix: csr_matrix,
    test_fraction: float = 0.2,
    min_items_for_test: int = 4,
    random_state: int = 42,
) -> tuple[csr_matrix, csr_matrix]:
    """Hold out a few known interactions per eligible user for evaluation."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    train = user_item_matrix.copy().tolil()
    test = csr_matrix(user_item_matrix.shape, dtype=np.float32).tolil()

    for user_idx in range(user_item_matrix.shape[0]):
        start, end = user_item_matrix.indptr[user_idx], user_item_matrix.indptr[user_idx + 1]
        item_indexes = user_item_matrix.indices[start:end]

        if len(item_indexes) < min_items_for_test:
            continue

        n_holdout = max(1, int(round(len(item_indexes) * test_fraction)))
        holdout_items = rng.choice(item_indexes, size=n_holdout, replace=False)

        for item_idx in holdout_items:
            test[user_idx, item_idx] = user_item_matrix[user_idx, item_idx]
            train[user_idx, item_idx] = 0.0

    train = train.tocsr()
    test = test.tocsr()
    train.eliminate_zeros()
    test.eliminate_zeros()
    return train, test
