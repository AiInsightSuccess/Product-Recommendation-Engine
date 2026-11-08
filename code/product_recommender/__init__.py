"""Beginner-friendly product recommendation helpers."""

from .data import (
    EVENT_WEIGHTS,
    build_user_item_matrix,
    load_sample_data,
    prepare_interaction_strengths,
    train_test_split_by_user,
)
from .evaluation import evaluate_recommendations
from .model import (
    apply_bm25_weighting,
    fit_als_model,
    get_recommendations_for_user,
    get_similar_items,
    run_optuna_search,
)

__all__ = [
    "EVENT_WEIGHTS",
    "apply_bm25_weighting",
    "build_user_item_matrix",
    "evaluate_recommendations",
    "fit_als_model",
    "get_recommendations_for_user",
    "get_similar_items",
    "load_sample_data",
    "prepare_interaction_strengths",
    "run_optuna_search",
    "train_test_split_by_user",
]
