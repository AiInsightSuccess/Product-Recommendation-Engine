# Code Folder

This folder contains optional standalone code for users who want to run the
recommendation workflow outside the notebook.

- `generate_sample_data.py`: recreates the sample CSV files. Note: this
  overwrites the CSV files in the main folder.
- `run_recommendation_engine.py`: runs the full recommendation workflow and
  saves outputs.
- `product_recommender/`: helper package used by the script and notebook.

From the package root folder, run:

```bash
python code/run_recommendation_engine.py
```
