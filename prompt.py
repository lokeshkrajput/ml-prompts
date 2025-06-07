
# === Agent Descriptions ===

eda_description = (
    "You are a financial data preparation expert. Your job is to prepare raw stock data "
    "for modeling. Focus on loading time series CSVs, ensuring key columns are present, "
    "handling duplicates via aggregation, converting types, setting a datetime index, and sorting. "
    "Ensure the output is clean, consistent, and usable for downstream feature engineering."
)

fe_description = (
    "You are a financial time series feature engineer. Your task is to extract predictive features "
    "from stock price data to help forecast the next day's return. Use domain-relevant techniques such as "
    "lag features, rolling statistics (mean, std, min, max), returns, and technical indicators like RSI, MACD, "
    "or Bollinger Bands. All features must be based strictly on past data — no lookahead bias. "
    "Ensure that the resulting DataFrames are numeric, NaN-free, and efficient to process. Optimize for memory and modeling performance."
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

eda_prompt_template = """Given the raw data paths: `train_csv` ('{train_csv}'), `val_csv` ('{val_csv}'), and `test_csv` ('{test_csv}'), generate a Python script named `EDA.py` that performs data cleaning and preparation.

Your script must:
1. Load the CSV files into pandas DataFrames.
2. Ensure columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] are present. If 'Adjusted Close' is missing, create it as a copy of 'Close'.
3. Convert 'Date' to datetime. Handle duplicate 'Date' rows by aggregating them (e.g., using groupby and common aggregation functions).
4. Set 'Date' as index and sort the data chronologically.
5. Ensure all numeric columns are correctly typed.
Return cleaned DataFrames for training, validation, and testing as `df_train`, `df_val`, and `df_test` respectively.
"""

fe_prompt_template = """Using the cleaned data from `EDA.py`, generate a script named `FEATURE.py` that performs feature engineering for stock return prediction.

Your script must:
1. Accept three DataFrames as input: `df_train`, `df_val`, and `df_test`, each with columns ['Open', 'High', 'Low', 'Close', 'Volume', 'Adjusted Close'] and datetime index.
2. For each DataFrame, generate the following features using only past data:
   - Log returns and percentage changes (1-day and multi-day)
   - Rolling statistics (mean, std, min, max) for windows of 5, 10, and 20 days
   - Technical indicators such as RSI, MACD, and Bollinger Bands
   - Lagged values of 'Close' and 'Volume' for 1, 2, 3, 5 days
3. Create a `target` column as the next day’s log return using shifted values.
4. Drop rows with any NaNs after feature creation.
5. Ensure all features are numeric and aligned with the target variable.
6. Return the processed DataFrames: `df_train_fe`, `df_val_fe`, and `df_test_fe`.

Use efficient operations (e.g., vectorized pandas ops, pd.concat) to minimize memory usage and fragmentation.
"""


model_prompt_template = """Using the features from `FEATURE.py`, generate a script named `MODEL.py` that trains a model for next-day return prediction.

Your script must:
1. Use `df_train_fe` and `df_val_fe` to train a regression model (e.g., LightGBM).
2. Predict the `target` column (next day log return).
3. Apply time-ordered splitting and early stopping using validation data.
4. Output trained model as `trained_model`, and compute evaluation metrics: MAE and RMSE.
5. Save model and metrics for use in the evaluation phase.
"""

eval_prompt_template = """Using the trained model from `MODEL.py`, generate a script named `EVAL.py` that evaluates prediction performance.

Your script must:
1. Load the model and use `df_test_fe` to make predictions.
2. Compute RMSE between predicted and true `target` values.
3. Save the RMSE to the specified output file in numeric format only.
4. Optionally include visual evaluation like prediction vs actual, if helpful.
"""
