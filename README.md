# Product Recommendation Engine (BM25 + ALS + Optuna)

Free Python code and sample data from
[AI Insights for Success](https://www.aiinsightsuccess.com/blank-2-1-1).
This repository contains a beginner-friendly Jupyter notebook, sample CSV files,
and standalone Python code for building a product recommendation engine.

## Get the Code

- **No Git?** Click the green **Code** button on this page, then
  **Download ZIP**, and unzip it.
- **Using Git:** `git clone https://github.com/AiInsightSuccess/Product-Recommendation-Engine.git`

The sample context is a generic online shop that sells household, personal care,
office, fitness, outdoor, kitchen, and pet products. No specific shop name is
used.

The method is an implicit-feedback matrix-factorization recommender:

1. Customer events such as views, clicks, basket adds, purchases, and reviews
   are converted into confidence scores.
2. The customer-product interaction data is stored as a sparse user-item matrix.
3. BM25 re-weighting reduces the dominance of very popular products and very
   active customers.
4. Alternating Least Squares (ALS) decomposes the interaction matrix into lower
   dimensional customer and product factor matrices.
5. Optuna searches model hyperparameters such as latent factors, regularization,
   ALS iterations, alpha, and BM25 settings.

## Recommended Beginner Setup

The easiest way to run the notebook is with Anaconda, because Anaconda installs
Python, Jupyter Notebook, and common data-analysis packages together.

Useful links:

- Download Anaconda: https://www.anaconda.com/download
- Jupyter Notebook help: https://jupyter.org/install

If you already use Python, install the required packages from a terminal opened
in this folder:

```bash
pip install -r requirements.txt
```

Python 3.10 to 3.12 is recommended. If `pip` cannot install the `implicit`
package on your computer (this can happen on Windows or with very new Python
versions), Anaconda users can install it with:

```bash
conda install -c conda-forge implicit optuna
```

If `implicit` or `optuna` is missing, the notebook still runs using a small
built-in teaching version of ALS and default settings, and it tells you so in
Cell 1. Results will differ from the full workflow.

## Files in This Folder

- `Product_Recommendation_Engine_Workbook.ipynb`: main notebook to run.
- `customer_product_interactions.csv`: sample customer-product events.
- `product_catalog.csv`: product names, categories, prices, stock status, and
  ratings used to label recommendations.
- `customer_segments.csv`: optional sample customer context. The ALS baseline
  does not use this file directly.
- `requirements.txt`: Python packages used by the notebook.
- `docs/`: webpage copy drafts.
- `code/`: reusable Python scripts for users who prefer code files, including
  `generate_sample_data.py`, which recreates the sample CSV files (it
  overwrites the CSV files in this folder).
- `outputs/`: folder where notebook and script results are saved.

## How to Run the Notebook

1. Download the repository (see *Get the Code* above).
2. Open Anaconda Navigator.
3. Click `Launch` under Jupyter Notebook.
4. In Jupyter, browse to this folder.
5. Open `Product_Recommendation_Engine_Workbook.ipynb`.
6. Run each notebook cell from top to bottom.
7. Review the printed results and the CSV files saved in `outputs/`.

If packages are missing, open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

## Optional Command-Line Run

Advanced users can run the same workflow from a terminal:

```bash
python code/run_recommendation_engine.py
```

## Input Data Format

The main input file is `customer_product_interactions.csv`.

Required columns:

- `user_id`: customer identifier.
- `item_id`: product identifier.
- `event_type`: event name, such as `view`, `click`, `add_to_cart`,
  `purchase`, or `review`.

Useful optional columns:

- `event_strength`: numeric value for the event. If omitted, the code uses the
  default weights in `code/product_recommender/data.py`.
- `quantity`: purchase quantity or `1` for non-purchase events.
- `unit_price`: product price at the time of interaction.
- `interaction_datetime`: event timestamp.
- `session_id`: browsing or purchase session identifier.

## Main Outputs

After running the notebook or script, check the `outputs/` folder for:

- `aggregated_user_item_interactions.csv`
- `best_als_parameters.csv`
- `als_optuna_trials.csv` when Optuna is installed and the search runs
- `als_evaluation_metrics.csv`
- `als_user_level_metrics.csv`
- `sample_user_recommendations.csv`
- `sample_similar_items.csv`
- `recommendation_category_mix.png`
- `optuna_trial_history.png` when Optuna is installed and the search runs

## Notes for Beginners

ALS works by repeatedly decomposing a large sparse customer-product interaction
matrix into two smaller matrices: one for customers and one for products. It
alternates between fixing the product matrix while learning the customer matrix,
then fixing the customer matrix while learning the product matrix. This makes it
effective for large datasets where most customers interact with only a small
share of products.

This package is a teaching example. For a production recommendation system, add
stronger validation, data quality checks, retraining schedules, monitoring, and
cold-start logic for new customers and new products.
