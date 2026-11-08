"""BM25 weighting, ALS fitting, recommendations, and Optuna search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .data import MatrixMappings


def apply_bm25_weighting(user_item_matrix: csr_matrix, k1: float = 100.0, b: float = 0.8) -> csr_matrix:
    """Apply BM25 re-weighting to a sparse user-item matrix."""

    try:
        from implicit.nearest_neighbours import bm25_weight

        return bm25_weight(user_item_matrix.tocsr(), K1=k1, B=b).tocsr()
    except ImportError:
        return _local_bm25_weight(user_item_matrix, k1=k1, b=b)


def _local_bm25_weight(user_item_matrix: csr_matrix, k1: float, b: float) -> csr_matrix:
    """Small local BM25 implementation used only when `implicit` is unavailable."""

    matrix = user_item_matrix.tocoo(copy=True).astype(np.float32)
    if matrix.nnz == 0:
        return matrix.tocsr()

    row_lengths = np.asarray(user_item_matrix.sum(axis=1)).ravel()
    average_length = max(float(row_lengths[row_lengths > 0].mean()), 1e-9)

    item_frequency = np.bincount(matrix.col, minlength=user_item_matrix.shape[1])
    idf = np.log((user_item_matrix.shape[0] + 1.0) / (item_frequency + 0.5))
    idf = np.maximum(idf, 0.0)

    length_norm = 1.0 - b + b * row_lengths[matrix.row] / average_length
    denominator = matrix.data + k1 * length_norm
    matrix.data = matrix.data * (k1 + 1.0) / denominator * idf[matrix.col]
    matrix.eliminate_zeros()
    return matrix.tocsr()


@dataclass
class SimpleImplicitALS:
    """Small educational implicit-feedback ALS fallback for the sample dataset."""

    factors: int = 32
    regularization: float = 0.05
    alpha: float = 40.0
    iterations: int = 10
    random_state: int = 42

    def fit(self, user_items: csr_matrix, show_progress: bool = False) -> "SimpleImplicitALS":
        rng = np.random.default_rng(self.random_state)
        self.user_factors = rng.normal(0, 0.01, size=(user_items.shape[0], self.factors))
        self.item_factors = rng.normal(0, 0.01, size=(user_items.shape[1], self.factors))

        item_users = user_items.T.tocsr()
        for iteration in range(self.iterations):
            self.user_factors = self._solve_factors(user_items, self.item_factors)
            self.item_factors = self._solve_factors(item_users, self.user_factors)
            if show_progress:
                print(f"Finished fallback ALS iteration {iteration + 1}/{self.iterations}")

        return self

    def _solve_factors(self, interactions: csr_matrix, fixed_factors: np.ndarray) -> np.ndarray:
        n_rows = interactions.shape[0]
        factors = np.zeros((n_rows, self.factors), dtype=np.float64)
        gram = fixed_factors.T @ fixed_factors
        identity = np.eye(self.factors)

        for row_idx in range(n_rows):
            start, end = interactions.indptr[row_idx], interactions.indptr[row_idx + 1]
            indexes = interactions.indices[start:end]
            values = interactions.data[start:end]

            a_matrix = gram + self.regularization * identity
            b_vector = np.zeros(self.factors, dtype=np.float64)

            for item_idx, value in zip(indexes, values):
                factor = fixed_factors[item_idx]
                confidence = 1.0 + self.alpha * float(value)
                a_matrix += (confidence - 1.0) * np.outer(factor, factor)
                b_vector += confidence * factor

            factors[row_idx] = np.linalg.solve(a_matrix, b_vector)

        return factors

    def recommend(
        self,
        userid: int,
        user_items: csr_matrix,
        N: int = 10,
        filter_already_liked_items: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = self.item_factors @ self.user_factors[userid]
        if filter_already_liked_items:
            already_seen = user_items.indices[user_items.indptr[0] : user_items.indptr[1]]
            scores = scores.copy()
            scores[already_seen] = -np.inf

        top_ids = np.argsort(scores)[::-1][:N]
        return top_ids, scores[top_ids]

    def similar_items(self, itemid: int, N: int = 10) -> tuple[np.ndarray, np.ndarray]:
        item_vector = self.item_factors[itemid]
        norms = np.linalg.norm(self.item_factors, axis=1) * max(np.linalg.norm(item_vector), 1e-12)
        scores = (self.item_factors @ item_vector) / np.maximum(norms, 1e-12)
        top_ids = np.argsort(scores)[::-1][:N]
        return top_ids, scores[top_ids]


def fit_als_model(
    weighted_user_items: csr_matrix,
    factors: int = 32,
    regularization: float = 0.05,
    alpha: float = 40.0,
    iterations: int = 12,
    random_state: int = 42,
    show_progress: bool = False,
) -> Any:
    """Fit an ALS model using `implicit`, with a small local fallback."""

    try:
        from implicit.als import AlternatingLeastSquares

        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            random_state=random_state,
        )
        model.fit(weighted_user_items, show_progress=show_progress)
        model.source_library = "implicit"
        return model
    except ImportError:
        model = SimpleImplicitALS(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            random_state=random_state,
        )
        model.fit(weighted_user_items, show_progress=show_progress)
        model.source_library = "local educational ALS fallback"
        return model


def get_recommendations_for_user(
    model: Any,
    user_id: str,
    mappings: MatrixMappings,
    train_user_items: csr_matrix,
    catalog: pd.DataFrame,
    n_recommendations: int = 10,
) -> pd.DataFrame:
    """Return a tidy recommendation table for one original user ID."""

    if user_id not in mappings.user_to_index:
        raise ValueError(f"Unknown user_id: {user_id}")

    user_idx = mappings.user_to_index[user_id]
    item_indexes, scores = model.recommend(
        user_idx,
        train_user_items[user_idx],
        N=n_recommendations,
        filter_already_liked_items=True,
    )

    result = pd.DataFrame(
        {
            "rank": np.arange(1, len(item_indexes) + 1),
            "item_id": [mappings.index_to_item[int(idx)] for idx in item_indexes],
            "recommendation_score": scores,
        }
    )
    return result.merge(catalog, on="item_id", how="left")


def get_similar_items(
    model: Any,
    item_id: str,
    mappings: MatrixMappings,
    catalog: pd.DataFrame,
    n_items: int = 10,
) -> pd.DataFrame:
    """Return products with latent factors similar to one selected product."""

    if item_id not in mappings.item_to_index:
        raise ValueError(f"Unknown item_id: {item_id}")

    item_idx = mappings.item_to_index[item_id]
    item_indexes, scores = model.similar_items(item_idx, N=n_items)
    result = pd.DataFrame(
        {
            "rank": np.arange(1, len(item_indexes) + 1),
            "item_id": [mappings.index_to_item[int(idx)] for idx in item_indexes],
            "similarity_score": scores,
        }
    )
    return result.merge(catalog, on="item_id", how="left")


def run_optuna_search(
    train_user_items: csr_matrix,
    test_user_items: csr_matrix,
    n_trials: int = 20,
    k: int = 10,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Use Optuna to search BM25 and ALS hyperparameters."""

    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for hyperparameter searching. "
            "Install the package with: pip install -r requirements.txt"
        ) from exc

    from .evaluation import evaluate_recommendations

    def objective(trial: optuna.Trial) -> float:
        params = {
            "factors": trial.suggest_categorical("factors", [16, 24, 32, 48, 64]),
            "regularization": trial.suggest_float("regularization", 1e-3, 0.3, log=True),
            "alpha": trial.suggest_float("alpha", 5.0, 80.0, log=True),
            "iterations": trial.suggest_int("iterations", 5, 20),
            "bm25_k1": trial.suggest_float("bm25_k1", 20.0, 180.0),
            "bm25_b": trial.suggest_float("bm25_b", 0.2, 1.0),
        }

        weighted_train = apply_bm25_weighting(
            train_user_items, k1=params["bm25_k1"], b=params["bm25_b"]
        )
        model = fit_als_model(
            weighted_train,
            factors=params["factors"],
            regularization=params["regularization"],
            alpha=params["alpha"],
            iterations=params["iterations"],
            random_state=random_state,
            show_progress=False,
        )
        metrics, _ = evaluate_recommendations(model, train_user_items, test_user_items, k=k)
        return float(metrics["mean_recall_at_k"])

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_df = study.trials_dataframe()
    return dict(study.best_params), trials_df
