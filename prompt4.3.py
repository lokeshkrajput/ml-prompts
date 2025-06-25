# ==============================================================
# Expert Agent Prompt Templates – FULL SCRIPT (v2-BULLET-ALT)
# ==============================================================

# -----------------------------------------------------------------
# SHARED CONSTANTS (keep these exactly the same for every agent)
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
Begin EVERY generated script with DEPENDENCY_CHECK exactly as provided above.  
Wrap all runnable code inside main() and call it via if __name__ == "__main__".  
Parse CLI flags with argparse; exit on unknown arguments.  
Seed ALL random number generators with 42.  
Write artefacts to ./artifacts (mkdir if missing).  
After successful completion print the single word DONE followed by a one-line JSON object, then STOP.  
Finish every assistant response with {SENTINEL}
"""

# ==============================================================
# 1 ▸ EDA AGENT
# ==============================================================

eda_description = """
Role    : Senior Data-Profiling Analyst  
Mission : Clean the raw train/val/test CSV splits (deduplicate dates, set ordered index),  
          write the cleaned files to disk, and emit a deterministic DONE status for downstream agents.
"""

eda_prompt_template = f"""
{GLOBAL_RULES}

INSTRUCTIONS  
1. Return **EDA.py** inside ```python``` fences.  
   - Accept flags --train_path, --val_path, --test_path  
   - For each split:  
       - Deduplicate identical Date rows (numerics → mean, categoricals → first)  
       - Set Date as index and sort ascending  
   - Save to artifacts/train_clean.csv, val_clean.csv, test_clean.csv  
   - Print exactly:  
       DONE {{ "artefact_paths": "artifacts/train_clean.csv;artifacts/val_clean.csv;artifacts/test_clean.csv" }}  
       STOP  
2. Return a ```bash``` block that:  
       chmod +x EDA.py  
       ./EDA.py --train_path "{{train_path}}" --val_path "{{val_path}}" --test_path "{{test_path}}"  
3. End the entire assistant message with {SENTINEL}  
{SENTINEL}
"""

# ==============================================================
# 2 ▸ FEATURE-ENGINEERING AGENT
# ==============================================================

fe_description = """
Role    : Feature-Engineering Architect  
Mission : Transform cleaned splits into model-ready features using lags, rolling statistics,  
          and common technical indicators, then save one parquet per split and signal DONE.
"""

fe_prompt_template = f"""
{GLOBAL_RULES}

INSTRUCTIONS  
1. Return **FEATURE.py** in ```python``` fences.  
   - Accept --train_path, --val_path, --test_path (clean CSVs)  
   - For every split:  
       - Generate lag features for 1, 2, 3, 5 days  
       - Rolling mean / std / min / max over 5, 10, 20-day windows  
       - If columns permit, add RSI, MACD, Bollinger Bands Width  
       - Drop rows introduced with NaNs and cast all to numeric  
   - Save to artifacts/train_features.parquet, val_features.parquet, test_features.parquet (snappy)  
   - Print:  
       DONE {{ "feature_paths": "artifacts/train_features.parquet;artifacts/val_features.parquet;artifacts/test_features.parquet" }}  
       STOP  
2. Return a ```bash``` block:  
       chmod +x FEATURE.py  
       ./FEATURE.py --train_path "{{train_clean}}" --val_path "{{val_clean}}" --test_path "{{test_clean}}"  
3. End with {SENTINEL}  
{SENTINEL}
"""

# ==============================================================
# 3 ▸ MODELLING / AUTOML AGENT
# ==============================================================

model_description = """
Role    : AutoML Optimisation Scientist  
Mission : Use Optuna to tune several ensemble regressors, retrain the best model on train + val,  
          and store the reproducible model artefact.
"""

model_prompt_template = f"""
{GLOBAL_RULES}

INSTRUCTIONS  
1. Return **MODEL.py** in ```python``` fences.  
   - Accept --train_path, --val_path, --test_path (feature parquets)  
   - Expect column "target" in each split  
   - Build an Optuna study (50 trials, 300 s timeout) exploring:  
       - GradientBoostingRegressor  
       - RandomForestRegressor  
       - HistGradientBoostingRegressor  
       - LGBMRegressor  
       - XGBRegressor  
       - CatBoostRegressor  
   - Objective: minimise RMSE on the validation split (no shuffling, time-aware)  
   - Retrain best hyper-params on train + val, save to artifacts/best_model.joblib  
   - Print:  
       DONE {{ "best_rmse": <float4>, "best_algo": "<name>" }}  
       STOP  
2. Return a ```bash``` block:  
       chmod +x MODEL.py  
       ./MODEL.py --train_path "{{train_feats}}" --val_path "{{val_feats}}" --test_path "{{test_feats}}"  
3. End with {SENTINEL}  
{SENTINEL}
"""

# ==============================================================
# 4 ▸ EVALUATION AGENT
# ==============================================================

eval_description = """
Role    : Reliability Metrics Auditor  
Mission : Score the persisted model on the held-out test split, archive RMSE/MAE/R²,  
          and output a DONE status for reporting.
"""

eval_prompt_template = f"""
{GLOBAL_RULES}

INSTRUCTIONS  
1. Return **EVALUATE.py** in ```python``` fences.  
   - Accept --model_path and --test_path  
   - Compute RMSE, MAE, R² on the test split  
   - Save metrics to artifacts/metrics.json (include UTC ISO timestamp)  
   - Print:  
       DONE {{ "metrics_path": "artifacts/metrics.json" }}  
       STOP  
2. Return a ```bash``` block:  
       chmod +x EVALUATE.py  
       ./EVALUATE.py --model_path artifacts/best_model.joblib --test_path "{{test_feats}}"  
3. End with {SENTINEL}  
{SENTINEL}
"""
