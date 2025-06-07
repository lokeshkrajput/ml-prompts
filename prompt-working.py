
# === Agent Descriptions ===

eda_description = (
    "You are a financial data preparation expert. Your job is to prepare raw stock data "
    "for modeling. Focus on loading time series CSVs, ensuring key columns are present, "
    "handling duplicates via aggregation, converting types, setting a datetime index, and sorting. "
    "Ensure the output is clean, consistent, and usable for downstream feature engineering."
)

fe_description = (
    "You are a time series feature engineer. Extract predictive, leakage-free features from stock price data "
    "to forecast the next day’s return. Use lag values, rolling stats, and technical indicators. Output must be numeric, clean, and efficient."
)

model_description = (
    "You are a time series modeling expert. Build a regression model to predict the next day’s stock return "
    "using engineered features. Use efficient libraries like LightGBM. Apply time-series splits, avoid data leakage, "
    "and prioritize speed and robustness. Include early stopping during training to prevent overfitting."
)

eval_description = (
    "You are an evaluation specialist. Your job is to assess a pre-trained stock return model using test data. "
    "Evaluate predictions, compute RMSE, and summarize performance. Save the RMSE in the required format for submission."
)

# === Agent Task Prompt Templates ===

eda_prompt_template = """Three input CSVs are located at:
TRAIN = '{train_csv}'
VAL   = '{val_csv}'
TEST  = '{test_csv}'

Run the EDA tasks on these files and save your script as EDA.py.

Ensure script must:
1. Handle duplicate 'Date' rows by aggregating them (e.g., using groupby and common aggregation mechanisms).
2. Set Date column as index and sort the data chronologically.
3. Save the EDA analysis/transformations for each split as train_clean.csv, val_clean.csv, test_clean.csv in local directory.
"""

fe_prompt_template = """The clean splits are train_clean.csv, val_clean.csv, test_clean.csv in the working directory.

Generate and execute a script named 'FEATURE.py' which engineers features per spec and save
train_features.csv, val_features.csv, test_features.csv.

1. Add lag features (1, 2, 3, 5 days) and rolling stats (mean, std, min, max) for 5, 10, 20-day windows.
2. Include indicators like RSI, MACD, Bollinger Bands if useful.
3. Create 'target' as next day's log return.
4. Drop rows with NaNs and ensure all features are numeric.

Use only past data. Optimize for performance and avoid feature leakage.
"""

model_prompt_template = """Features files are train_features.csv, val_features.csv, test_features.csv
Generate an executable script named 'MODEL.py' that trains a model for next-day return prediction.

Before writing the script:
1. Check for any incompatible version of LightGBM and re-install compatible one, if required
2. Ensure the python code is compatible to the installed LightGBM library by referring LightGBM API specifications.
3. Predict the right columns in feature files for operations

Ensure script must:
- Train using regression model (e.g., LightGBM)
- Predict the 'target' column (next day log return)
- Apply time-ordered splitting and early stopping using validation data
- Output trained model as model.pkl and compute evaluation metrics: MAE and RMSE
- Save model and metrics for use in the evaluation phase
"""

eval_prompt_template = """Trained model is 'model.pkl', generate and execute script named 'EVAL.py'.

Ensure script must:
- Evaluate model.pkl on test_features.csv
- Compute RMSE in it, and write the item to MSFT_Score.txt
- Optionally include visual evaluation like prediction vs actual

Make sure the test_features.csv contains a column named 'target'. If not, match the correct one using training files.
"""
