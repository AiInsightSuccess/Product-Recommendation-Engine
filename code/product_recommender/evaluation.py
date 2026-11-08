"""Simple recommender evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def evaluate_recommendations(
    model: Any,
    train_user_items: csr_matrix,
    test_user_items: csr_matrix,
    k: int = 10,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Calculate precision@k, recall@k, and hit-rate@k on held-out items."""

    user_rows = []

    for user_idx in range(test_user_items.shape[0]):
        test_start = test_user_items.indptr[user_idx]
        test_end = test_user_items.indptr[user_idx + 1]
        true_items = set(test_user_items.indices[test_start:test_end])

        if not true_items:
            continue

        recommended_items, _ = model.recommend(
            user_idx,
            train_user_items[user_idx],
            N=k,
            filter_already_liked_items=True,
        )
        recommended_set = set(map(int, recommended_items))
        hits = len(true_items.intersection(recommended_set))

        user_rows.append(
            {
                "user_index": user_idx,
                "test_items": len(true_items),
                "hits_at_k": hits,
                "precision_at_k": hits / k,
                "recall_at_k": hits / len(true_items),
                "hit_rate_at_k": float(hits > 0),
            }
        )

    user_metrics = pd.DataFrame(user_rows)
    if user_metrics.empty:
        return {
            "evaluated_users": 0,
            "mean_precision_at_k": 0.0,
            "mean_recall_at_k": 0.0,
            "mean_hit_rate_at_k": 0.0,
        }, user_metrics

    summary = {
        "evaluated_users": int(len(user_metrics)),
        "mean_precision_at_k": float(user_metrics["precision_at_k"].mean()),
        "mean_recall_at_k": float(user_metrics["recall_at_k"].mean()),
        "mean_hit_rate_at_k": float(user_metrics["hit_rate_at_k"].mean()),
    }
    return summary, user_metrics
