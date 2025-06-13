# ─── EDA (force modern ffill / bfill) ─────────────────────────────────────
eda_description = """
You are a Python best-practices expert and EDA specialist.

Required behaviour
- Never call DataFrame.fillna(method=…).  Instead:
      df = df.ffill().bfill()
- All other steps stay as before (date parsing if present, duplicates, down-cast,
  drop >30 % missing cols, save *_clean.csv, print schema, sys.exit(1) on error).
"""

eda_prompt_template = """
TRAIN = {train_csv}
VAL   = {val_csv}
TEST  = {test_csv}

Create EDA.py that follows every line in the description.  **Insert an assert
after cleaning that .fillna(  does NOT appear in the script itself** (so the LLM
cannot sneak it back in).
"""

# ─── FEATURE ENGINEERING (robust price detection) ────────────────────────
fe_description = """
You are a Python best-practices expert and feature-engineering specialist.

Price-series selection logic
1. If a column named Close (case-insensitive) exists → use it.
2. Else if Adjusted_close / Adj_Close exists → use it.
3. Else if exactly one numeric column appears aside from Volume or target → use that.
4. Otherwise print an error and sys.exit(1).

Technical indicators
- First try import talib
- Else try import pandas_ta
- Else compute indicators manually (vectorised pandas / numpy).  No pip installs.

Features to create remain: price & volume lags 1,2,3,5; rolling mean/std/min/max
over 5,10,20; RSI-14, MACD 12-26-9, BB 20-2; calendar d-o-w, month, year.

Output files train_features.csv, val_features.csv, test_features.csv must share
identical column order & dtype; drop rows that became NaN; print shape + memory;
sys.exit(1) on fatal error.
"""

fe_prompt_template = """
Generate FEATURE.py that exactly implements the description.  **Add an assert
that the string '\"price\"' does NOT appear anywhere in the script** so the agent
cannot hard-code that name again.
"""


# ========== 3. MODELLING AGENT ==========
model_description = """
You are a Python best-practices expert and time-series modelling specialist.
Build the most accurate yet efficient next-day log-return model.

Inputs are train_features.csv, val_features.csv and test_features.csv with the
target column named target.  A file called model-type may contain one of
lightgbm, xgboost, catboost, rf, enet or linear; default to lightgbm if the file
is missing or invalid.  Run on CPU only, keep total runtime below fifteen
minutes and ensure model.pkl is no larger than seventy-five megabytes.  At the
end print a single JSON line containing best_cv_rmse, val_rmse and val_mae.
Exit with sys.exit(1) on fatal error.
"""

model_prompt_template = """
Create MODEL.py that performs these actions:

1. import only numpy, pandas, sklearn, lightgbm, xgboost, catboost, optuna,
   joblib, pathlib, warnings, json and sys; suppress warnings and set a module
   constant RANDOM_STATE equal to forty-two.

2. load the three feature CSVs, sort by date, down-cast floats to float32 and
   verify the schemas are identical.

3. build a TimeSeriesSplit with five splits whose test_size equals len(val_df);
   wrap StandardScaler(with_mean=False) and the model inside a sklearn Pipeline
   to prevent leakage.

4. choose the algorithm from model-type or default to lightgbm; for tree models
   run an Optuna study with at most forty trials or nine-hundred seconds and use
   early_stopping_rounds equal to fifty; for linear or enet fit ElasticNetCV
   with the same splitter.

5. retrain the best model on the union of train and val and save it to
   model.pkl compressed with xz.

6. predict on val, compute RMSE and MAE and print the required JSON summary;
   if the model is tree-based also save feature_importance.png.

7. wrap everything in a main guard and call sys.exit(1) if any uncaught
   exception occurs.
"""

# ========== 4. EVALUATION AGENT ==========
eval_description = """
You are a Python best-practices expert and model-evaluation specialist.  Assess
the frozen model on unseen data.

Tasks: load model.pkl and test_features.csv; predict the target values; compute
RMSE, MAE and MAPE; write those three numbers, one per line, to
MSFT_score.txt; optionally save a scatter plot preds_vs_actual.png if
matplotlib is available; print a single JSON line with rmse, mae, mape and
n_test_rows; call sys.exit(1) if any required file is missing.
"""

eval_prompt_template = """
Create EVAL.py that carries out every step listed in the description.
"""
