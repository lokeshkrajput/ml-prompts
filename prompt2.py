
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
    TRAIN = '{{train_csv}}'
    VAL   = '{{val_csv}}'
    TEST  = '{{test_csv}}'

Run the EDA tasks on these files and save your script as EDA.py.

Ensure script must:
1. Handle duplicate 'Date' rows by aggregating them (e.g., using groupby and common aggregation mechanism and functions).
2. Set Date column as index and sort the data chronologically.
3. Save the EDA analysis/transformations for each split as train_clean.csv, val_clean.csv, test_clean.csv in local directory.
"""

fe_prompt_template = """The clean splits are train_clean.csv, val_clean.csv, test_clean.csv in the working directory.

Generate and execute a script named 'FEATURE.py' which engineer features per spec and save
train_features.csv, val_features.csv, test_features.csv.

1. Add lag features (1, 2, 3, 5 days) and rolling stats (mean, std, min, max) for 5, 10, 20-day windows.
2. Include indicators like RSI, MACD, Bollinger Bands if useful.
3. Create 'target' as next day's log return.
4. Drop rows with NaNs and ensure all features are numeric.

Use only past data. Optimize for performance and avoid feature leakage.
"""

model_prompt_template = """The feature files are: train_features.csv, val_features.csv, test_features.csv.

Generate a Python script named `MODEL.py` that trains a LightGBM regression model to predict next-day log return.

Before generating the script:
1. Check if LightGBM is installed and compatible.
2. Ensure the 'target' column exists and is correctly typed in the training files.

Your script must:
- Train using LightGBM's `train()` function with proper time-ordered data.
- Use `valid_sets` with `early_stopping()` via callbacks.
- Avoid using unsupported arguments like `early_stopping_rounds` or `verbose_eval`.
- Save the trained model as model.pkl.
- Print evaluation metrics: MAE and RMSE.

Validate the script syntax before execution and ensure compatibility with LightGBM ≥ 4.0.
"""


eval_prompt_template = """Trained model is 'model.pkl', generate and execute script named 'EVAL.py'.

Ensure script must:
1. Evaluate model.pkl on test_features.csv.
2. Compute RMSE if present column is 'target', else match it by comparing with train_features.csv.
3. Write the score to MSFT_Score.txt as a single float.
4. Optionally include visual evaluation like prediction vs actual, if helpful.
"""
