# Website Page Instructions

Use the text below on the webpage where users download the product
recommendation engine folder.

## Product Recommendation Engine Notebook

Download the complete folder below to run the example product recommendation
engine on your own computer. The folder includes a Jupyter notebook, sample CSV
files, reusable Python code, and an outputs folder.

### Download

Download the ZIP file:

`Product_Recommendation_Engine_Package.zip`

After downloading, unzip the folder before opening the notebook.

### What You Need

For basic users, we recommend installing Anaconda. Anaconda includes Python,
Jupyter Notebook, and many common data-analysis packages.

- Download Anaconda: https://www.anaconda.com/download
- Jupyter Notebook help: https://jupyter.org/install

If you already use Python, install the required packages with:

```bash
pip install -r requirements.txt
```

### How to Run the Notebook

1. Download and unzip `Product_Recommendation_Engine_Package.zip`.
2. Install Anaconda if Python or Jupyter Notebook is not already installed.
3. Open Anaconda Navigator.
4. Click **Launch** under Jupyter Notebook.
5. In the Jupyter browser window, open the unzipped folder.
6. Open `Product_Recommendation_Engine_Workbook.ipynb`.
7. Run the notebook cells from top to bottom.

### What Files Are Included

- `Product_Recommendation_Engine_Workbook.ipynb`: the main notebook.
- `customer_product_interactions.csv`: sample customer-product events.
- `product_catalog.csv`: product information used to label recommendations.
- `customer_segments.csv`: optional customer context.
- `requirements.txt`: package list for users not using Anaconda.
- `README.md`: detailed local instructions.
- `WEBPAGE_COPY.md`: short webpage copy.
- `code/`: standalone Python scripts.
- `outputs/`: location where notebook results are saved.

### What the Notebook Does

The notebook has six main parts:

1. **Import packages**  
   Loads pandas, NumPy, SciPy, Matplotlib, the local helper code, Optuna, and
   the ALS recommendation tools.

2. **Load the sample CSV files**  
   Reads customer-product interactions, product catalog information, and
   optional customer segment context.

3. **Create the user-item matrix**  
   Converts event rows into weighted implicit-feedback scores and builds a
   sparse customer-product matrix.

4. **Split the data for evaluation**  
   Holds out some known interactions per customer, so the model can be measured
   on products it has not seen during training.

5. **Train BM25 plus ALS**  
   Applies BM25 re-weighting and trains an Alternating Least Squares
   matrix-factorization model.

6. **Search hyperparameters with Optuna**  
   Uses Optuna to test different ALS and BM25 settings and selects the set that
   gives the best recall@10 on held-out interactions.

### Main Outputs

The notebook creates:

- `outputs/aggregated_user_item_interactions.csv`
- `outputs/best_als_parameters.csv`
- `outputs/als_optuna_trials.csv` when Optuna is installed and the search runs
- `outputs/als_evaluation_metrics.csv`
- `outputs/als_user_level_metrics.csv`
- `outputs/sample_user_recommendations.csv`
- `outputs/sample_similar_items.csv`
- `outputs/recommendation_category_mix.png`
- `outputs/optuna_trial_history.png` when Optuna is installed and the search runs

### Input File Format

`customer_product_interactions.csv` should contain:

- `user_id`
- `item_id`
- `event_type`
- `event_strength`
- `quantity`
- `unit_price`
- `interaction_datetime`
- `session_id`

The most important columns are `user_id`, `item_id`, and `event_type`.

### Common Problems

- **Jupyter does not open:** install Anaconda and launch Jupyter Notebook from
  Anaconda Navigator.
- **Package missing:** run `pip install -r requirements.txt`.
- **File not found:** keep the notebook and CSV files in the same folder.
- **No recommendations returned:** check that each customer has several
  interactions and that product IDs in the interaction file match
  `product_catalog.csv`.
- **Optuna takes too long:** reduce `N_OPTUNA_TRIALS` in the notebook.
