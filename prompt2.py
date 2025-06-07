
# === Agent Descriptions ===

eda_description = (
    "You are a financial data preparation expert. Prepare raw stock data for modeling by cleaning and validating it. "
    "Ensure critical columns exist, remove duplicates using aggregation, convert types, and set a datetime index. "
    "Your output should be reliable, well-structured, and ready for high-precision feature extraction."
)

fe_description = (
    "You are a time series feature engineer. Build predictive, leakage-free features for next-day return forecasting. "
    "Use lag values, rolling stats, and technical indicators. All features must be strictly based on past data. "
    "Ensure the DataFrames are numeric, clean, and optimized for model performance in a competitive setting."
)

model_description = (
    "You are a financial modeling expert. Build a high-precision model to predict next-day returns, minimizing RMSE to e-6 level. "
    "Use LightGBM with time-aware validation, regularization, and tuned hyperparameters. Avoid deprecated LightGBM parameters. "
    "Drop low-importance features if helpful and output a robust, optimized model for evaluation."
)

eval_description = (
    "You are a performance evaluator for financial models. Use the test set to generate predictions and calculate RMSE "
    "against true values. Save the RMSE in the required format for submission. Visualize performance if useful."
)

# === Agent Task Prompt Templates ===

eda_prompt_template = """Given the raw data paths: `train_csv` ('{train_csv}'), `val_csv` ('{val_csv}'), and `test_csv` ('{test_csv}'), generate a Python script named `EDA.py`.

Your script must:
1. Load CSVs into pandas DataFrames.
2. Ensure columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] exist. If 'Adjusted Close' is missing, create it from 'Close'.
3. Convert 'Date' to datetime. Aggregate duplicate dates using functions like first, max, min, last, sum.
4. Set 'Date' as index and sort by time.
5. Convert all financial columns to appropriate numeric types.
Return cleaned DataFrames: `df_train`, `df_val`, `df_test` ready for precise feature engineering.
"""

fe_prompt_template = """Create `FEATURE.py` to generate features from cleaned DataFrames: `df_train`, `df_val`, `df_test`.

Your script must:
1. Add lag features (1, 2, 3, 5 days) and rolling stats (mean, std, min, max) with 5, 10, 20-day windows.
2. Add technical indicators like RSI, MACD, and Bollinger Bands where beneficial.
3. Create `target` as next day's log return using shifted 'Adjusted Close'.
4. Drop rows with NaNs post feature generation. Ensure all features are numeric.
5. Return `df_train_fe`, `df_val_fe`, `df_test_fe`.

Use only past data. Design for accuracy, efficiency, and competition-level robustness.
"""

model_prompt_template = """Using features from `FEATURE.py`, generate `MODEL.py` to train a regression model targeting e-6 RMSE.

Steps:
1. Use LightGBM with `df_train_fe` and `df_val_fe`.
2. Predict `target` using time-aware splitting.
3. Use early stopping via callbacks (e.g., lgb.early_stopping).
4. Avoid deprecated arguments like early_stopping_rounds or verbose_eval.
5. Tune: learning_rate, num_leaves, max_depth, lambda_l1, lambda_l2.
6. Optionally drop low-importance features.
7. Save model as `trained_model.pkl` and print MAE, RMSE.
"""

eval_prompt_template = """Using `trained_model.pkl`, generate `EVAL.py` to evaluate prediction accuracy.

Tasks:
1. Load the model.
2. Predict using `test_features.csv` (ensure it contains a column named `target` or match it using training files).
3. Compute RMSE between predicted values and true labels.
4. Save RMSE to `MSFT_Score.txt` as a single number (no extra formatting).
5. (Optional) Visualize prediction vs actual and residuals.

Ensure the correct column is used for true values in `test_features.csv`.
"""
