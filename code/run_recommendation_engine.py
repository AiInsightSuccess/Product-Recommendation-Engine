"""Run the full sample recommendation workflow from the command line."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from product_recommender import (
    apply_bm25_weighting,
    build_user_item_matrix,
    evaluate_recommendations,
    fit_als_model,
    get_recommendations_for_user,
    get_similar_items,
    load_sample_data,
    prepare_interaction_strengths,
    run_optuna_search,
    train_test_split_by_user,
)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    matplotlib_cache_dir = output_dir / ".matplotlib"
    matplotlib_cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

    import matplotlib.pyplot as plt

    interactions, catalog = load_sample_data(base_dir)
    prepared = prepare_interaction_strengths(interactions)
    user_item_matrix, aggregated, mappings = build_user_item_matrix(prepared)
    train_matrix, test_matrix = train_test_split_by_user(user_item_matrix)

    aggregated.to_csv(output_dir / "aggregated_user_item_interactions.csv", index=False)

    try:
        best_params, trials_df = run_optuna_search(
            train_matrix,
            test_matrix,
            n_trials=20,
            k=10,
            random_state=42,
        )
        trials_df.to_csv(output_dir / "als_optuna_trials.csv", index=False)
        print("Optuna best parameters:")
    except ImportError as exc:
        print(exc)
        print("Using beginner-safe default parameters so the rest of the workflow can run.")
        best_params = {
            "factors": 32,
            "regularization": 0.05,
            "alpha": 40.0,
            "iterations": 12,
            "bm25_k1": 100.0,
            "bm25_b": 0.8,
        }

    pd.DataFrame([best_params]).to_csv(output_dir / "best_als_parameters.csv", index=False)
    print(best_params)

    weighted_train = apply_bm25_weighting(
        train_matrix, k1=best_params["bm25_k1"], b=best_params["bm25_b"]
    )
    model = fit_als_model(
        weighted_train,
        factors=int(best_params["factors"]),
        regularization=float(best_params["regularization"]),
        alpha=float(best_params["alpha"]),
        iterations=int(best_params["iterations"]),
        random_state=42,
        show_progress=False,
    )

    metrics, user_metrics = evaluate_recommendations(model, train_matrix, test_matrix, k=10)
    pd.DataFrame([metrics]).to_csv(output_dir / "als_evaluation_metrics.csv", index=False)
    user_metrics.to_csv(output_dir / "als_user_level_metrics.csv", index=False)

    sample_users = list(mappings.user_to_index.keys())[:5]
    recommendation_tables = [
        get_recommendations_for_user(model, user_id, mappings, train_matrix, catalog, 10).assign(
            user_id=user_id
        )
        for user_id in sample_users
    ]
    all_recommendations = pd.concat(recommendation_tables, ignore_index=True)
    all_recommendations.to_csv(output_dir / "sample_user_recommendations.csv", index=False)

    category_counts = all_recommendations["category"].value_counts().sort_values()
    plt.figure(figsize=(8, 5))
    category_counts.plot(kind="barh", color="#287c76")
    plt.title("Category Mix in Sample Recommendations")
    plt.xlabel("Recommendation Count")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(output_dir / "recommendation_category_mix.png", dpi=150)
    plt.close()

    trials_path = output_dir / "als_optuna_trials.csv"
    if trials_path.exists():
        trials_df = pd.read_csv(trials_path)
        if "value" in trials_df.columns and "number" in trials_df.columns:
            plt.figure(figsize=(8, 4))
            plt.plot(trials_df["number"], trials_df["value"], marker="o", color="#455a64")
            plt.title("Optuna Trial History")
            plt.xlabel("Trial Number")
            plt.ylabel("Recall@10")
            plt.tight_layout()
            plt.savefig(output_dir / "optuna_trial_history.png", dpi=150)
            plt.close()

    sample_item = list(mappings.item_to_index.keys())[0]
    get_similar_items(model, sample_item, mappings, catalog, 10).to_csv(
        output_dir / "sample_similar_items.csv", index=False
    )

    print("Saved outputs in:", output_dir)
    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
