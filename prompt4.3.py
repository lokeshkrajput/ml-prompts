# ==============================================================
# Expert-Agent Prompt Templates  ·  “single-pass safe” version
# --------------------------------------------------------------
# • EDA prompt is pre-formatted with GLOBAL_RULES & SENTINEL only
#   but **leaves {train_csv}/{val_csv}/{test_csv} untouched**
#   so your controller can .format() them later without KeyErrors.
# • FEATURE / MODEL / EVALUATE use hard-coded artifact paths, so
#   no second formatting step is needed for them.
# ==============================================================

# -----------------------------------------------------------------
# SHARED CONSTANTS
# -----------------------------------------------------------------
SENTINEL = "<<<END>>>"

DEPENDENCY_CHECK = r"""\
### BEGIN_DEPENDENCY_CHECK
import importlib, subprocess, sys, importlib.metadata
pkgs = {
    "numpy":        ">=1.26,<3",
    "pandas":       ">=2.2,<3",
    "scikit-learn": ">=1.4,<2",
    "joblib":       ">=1.4,<2",
    "optuna":       ">=3.6,<4",
    "matplotlib":   ">=3.9,<4",
    "lightgbm":     ">=4.3,<5",
    "xgboost":      ">=2.0.3,<3",
    "catboost":     ">=1.2,<2",
}
def ensure_pkg(name, spec):
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        print(f"Installing {name}{spec} …", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", f"{name}{spec}"]
        )
for _n, _v in pkgs.items():
    ensure_pkg(_n, _v)
### END_DEPENDENCY_CHECK
"""

GLOBAL_RULES = f"""\
Use only stdlib, numpy, pandas, scikit-learn, lightgbm, xgboost, catboost, joblib, optuna, matplotlib.  
Begin EVERY generated script with DEPENDENCY_CHECK exactly as above.  
Wrap runnable code inside main() and call it with if __name__ == "__main__".  
Seed all RNGs with 42.  
Write artefacts to ./artifacts (mkdir if missing).  
When finished, print DONE followed by a one-line JSON object, then STOP.  
Finish every assistant response with {SENTINEL}
"""

# ==============================================================
# 1 ▸ EDA AGENT
# ==============================================================

eda_description = """
Role    : Senior Data-Profiling Analyst  
Mission : Clean the raw CSV splits, save cleaned files, emit DONE JSON.
"""

# double-brace the later placeholders so they survive this .format()
eda_prompt_template = """
{GLOBAL_RULES}

INSTRUCTIONS
1. Return **EDA.py** inside ```python``` fences.
   - Accept flags --train_path, --val_path, --test_path
   - For each split:
       - Deduplicate identical Date rows (numeric → mean, categorical → first)
       - Set Date as index and sort ascending
   - Save to artifacts/train_clean.csv, val_clean.csv, test_clean.csv
   - Print exactly:
       DONE {{ "artifact_paths": "artifacts/train_clean.csv;artifacts/val_clean.csv;artifacts/test_clean.csv" }}
       STOP
2. Return a ```bash``` block that:
       chmod +x EDA.py
       ./EDA.py --train_path "{{train_csv}}" --val_path "{{val_csv}}" --test_path "{{test_csv}}"
3. End with {SENTINEL}
{SENTINEL}
""".format(GLOBAL_RULES=GLOBAL_RULES, SENTINEL=SENTINEL)

# ==============================================================
# 2 ▸ FEATURE-ENGINEERING AGENT   (hard-coded inputs)
# ==============================================================

fe_description = """
Role    : Feature-Engineering Architect  
Mission : Generate lag / rolling / technical features and save Parquet outputs.
"""

fe_prompt_template = f"""
{GLOBAL_RULES}

INSTRUCTIONS
1. Return **FEATURE.py** in ```python``` fences.
   - Default paths inside script:
       train_clean = artifacts/train_clean.csv
       val_clean   = artifacts/val_clean.csv
       test_clean  = artifacts/test_clean.csv
   - For each split:
       - Create lags 1, 2, 3, 5 days
       - Rolling mean / std / min / max over 5, 10, 20-day windows
       - Add RSI, MACD, Bollinger Bands Width when possible
       - Drop rows with new NaNs, coerce all columns numeric
   - Save to artifacts/train_features.parquet, val_features.parquet, test_features.parquet (snappy)
   - Print:
       DONE {{ "feature_paths": "artifacts/train_features.parquet;artifacts/val_features.parquet;artifacts/test_features.parquet" }}
       STOP
2. Return a ```bash``` block:
       chmod +x FEATURE.py
       ./FEATURE.py
3. End with {SENTINEL}
{SENTINEL}
"""

# ==============================================================
# 3 ▸ MODEL / AUTOML AGENT    (hard-coded inputs)
# ==============================================================

model_description = """
Role    : AutoML Optimisation Scientist  
Mission : Tune ensemble regressors, retrain the best, persist the model.
"""

model_prompt_template = f"""
{GLOBAL_RULES}

INSTRUCTIONS
1. Return **MODEL.py** in ```python``` fences.
   - Default paths:
       train_feats = artifacts/train_features.parquet
       val_feats   = artifacts/val_features.parquet
       test_feats  = artifacts/test_features.parquet
   - Expect column "target" in all splits
   - Use Optuna (50 trials, timeout 300 s) to minimise RMSE on validation split
     among:
       GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor,
       LGBMRegressor, XGBRegressor, CatBoostRegressor
   - Retrain best model on train + val, save to artifacts/best_model.joblib
   - Print:
       DONE {{ "best_rmse": <float4>, "best_algo": "<name>" }}
       STOP
2. Return a ```bash``` block:
       chmod +x MODEL.py
       ./MODEL.py
3. End with {SENTINEL}
{SENTINEL}
"""

# ==============================================================
# 4 ▸ EVALUATION AGENT        (hard-coded inputs)
# ==============================================================

eval_description = """
Role    : Reliability Metrics Auditor  
Mission : Score the persisted model on test data and archive metrics.
"""

eval_prompt_template = f"""
{GLOBAL_RULES}

INSTRUCTIONS
1. Return **EVALUATE.py** in ```python``` fences.
   - Default paths:
       model_path = artifacts/best_model.joblib
       test_feats = artifacts/test_features.parquet
   - Compute RMSE, MAE, R² on the test split
   - Save to artifacts/metrics.json (include UTC ISO timestamp)
   - Print:
       DONE {{ "metrics_path": "artifacts/metrics.json" }}
       STOP
2. Return a ```bash``` block:
       chmod +x EVALUATE.py
       ./EVALUATE.py
3. End with {SENTINEL}
{SENTINEL}
"""

# ==============================================================
# HOW TO USE IN CONTROLLER
# --------------------------------------------------------------
# eda_prompt = eda_prompt_template.format(
#     train_csv="path/to/train.csv",
#     val_csv  ="path/to/val.csv",
#     test_csv ="path/to/test.csv"
# )
# fe_prompt  = fe_prompt_template        # no formatting needed
# model_prompt = model_prompt_template   # no formatting needed
# eval_prompt  = eval_prompt_template    # no formatting needed
#
# Then stream each prompt to its agent, parse the DONE line JSON,
# and move to the next step.
# ==============================================================
