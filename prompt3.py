# ============================ 1 ─── EDA AGENT ────────────────────────────
eda_description = """
You are a **Python best-practices expert** and EDA specialist.  
Clean raw stock-price data so downstream agents receive tidy, memory-efficient
CSV files.

Key steps
1. Read TRAIN / VAL / TEST CSVs (paths supplied in the template).  
2. If a column named “date” (case-insensitive) exists → parse to datetime,
   set as index, sort ascending; otherwise skip this step.      # <- guard added
3. Remove duplicate rows and columns with a single distinct value.  
4. Down-cast numerics to float32 / int32.  
5. Handle missing data:  
   • drop columns with > 30 % NaNs,  
   • forward-fill then back-fill the remaining gaps.  
6. Save `train_clean.csv`, `val_clean.csv`, `test_clean.csv`.  
7. Print a one-line schema summary (column name, dtype, n_rows) per split.  
Use try/except and **sys.exit(1)** on fatal errors.
"""

eda_prompt_template = """
Three input CSV paths  
TRAIN = {train_csv}  
VAL   = {val_csv}  
TEST  = {test_csv}

Write a script called **EDA.py** that implements the steps in the description
exactly and nothing more.
"""

# ====================== 2 ─── FEATURE-ENGINEERING AGENT ──────────────────
fe_description = """
You are a **Python best-practices expert** and time-series feature engineer.
Create leakage-free, numeric float32 features for next-day log-return
prediction.

Mandatory features
• Price & volume lags: 1 / 2 / 3 / 5 days  
• Rolling mean | std | min | max over 5 / 10 / 20 days  
• Technical indicators: RSI-14, MACD-(12,26,9), Bollinger Bands-(20,2)  
• Calendar: day_of_week, month, year

TA-Lib note  
Attempt **import talib**. If that fails, fall back to **pandas_ta**; if that
also fails, compute the indicators manually with pandas/numpy. *No* pip
install commands in the script—just log which method was chosen.

Output
• `train_features.csv`, `val_features.csv`, `test_features.csv`  
   – identical column order & dtype  
   – drop rows that became NaN after feature creation  
• Print shape and memory-usage for each file.  
Fatal errors must call **sys.exit(1)**.
"""

fe_prompt_template = """
Clean splits are already present as train_clean.csv, val_clean.csv,
test_clean.csv.

Generate **FEATURE.py** that fulfils the description above.
"""

# ========================= 3 ─── MODELLING AGENT ─────────────────────────
model_description = """
You are a **Python best-practices expert** and time-series modelling
specialist.  Build the most accurate yet efficient next-day log-return model.

Inputs          : train_features.csv, val_features.csv, test_features.csv  
Target column   : target  
Optional hint   : file “model-type” may contain lightgbm | xgboost | catboost |
                  rf | enet | linear   (default = lightgbm)  
Constraints     : CPU only, ≤ 15 min runtime, model.pkl ≤ 75 MB  
Metrics to show : best_cv_rmse, val_rmse, val_mae (single JSON line)  
On fatal error  : sys.exit(1)
"""

model_prompt_template = """
Create **MODEL.py** that:

1️⃣ Imports only needed parts of numpy, pandas, sklearn, lightgbm, xgboost,
   catboost, optuna, joblib, pathlib, warnings, json, sys.  
   – warnings.filterwarnings("ignore"), RANDOM_STATE = 42.

2️⃣ Loads the three CSVs, sorts by date, down-casts floats to float32, verifies
   identical schemas.

3️⃣ Sets up TimeSeriesSplit(n_splits=5, test_size=len(val_df)) and wraps
   StandardScaler(with_mean=False) + model in a Pipeline.

4️⃣ Chooses algorithm from model-type (else lightgbm).  
   • Tree models → Optuna (≤ 40 trials or 900 s, early_stopping_rounds=50).  
   • Linear / enet → ElasticNetCV with the same splitter.

5️⃣ Retrains best model on train + val, saves **model.pkl** compressed (xz).

6️⃣ Predicts on val, computes RMSE & MAE, prints a JSON dict
   {"best_cv_rmse":…, "val_rmse":…, "val_mae":…, "algorithm":…, "n_features":…}.  
   Saves feature_importance.png for tree models only.

7️⃣ Wraps everything in if __name__ == "__main__": and sys.exit(1) on failure.
"""

# ========================= 4 ─── EVALUATION AGENT ────────────────────────
eval_description = """
You are a **Python best-practices expert** and model-evaluation specialist.
Assess the frozen model on unseen data and write competition scores.

Tasks
1. Load model.pkl and test_features.csv.  
2. Predict target, compute RMSE, MAE, MAPE.  
3. Write the three numbers, one per line, to **MSFT_score.txt**.  
4. Optionally save preds_vs_actual.png scatter if matplotlib available.  
5. Print a JSON line {"rmse":…, "mae":…, "mape":…, "n_test_rows":…}.  
Use sys.exit(1) if any required file is missing.
"""

eval_prompt_template = """
Generate **EVAL.py** that performs the tasks in the description exactly.
"""
