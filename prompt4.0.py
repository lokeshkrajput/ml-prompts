# ---------------------------------------------------------------------
# SHARED CONSTANTS (unchanged except for the updated GLOBAL_RULES)
# ---------------------------------------------------------------------

SENTINEL = "###END_OF_AGENT_OUTPUT###"

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
- Use only: stdlib, numpy, pandas, scikit-learn, lightgbm, xgboost, catboost, \
joblib, optuna, matplotlib
- Begin every script with the DEPENDENCY_CHECK block exactly as shown
- Define ART_DIR = "artifacts" once at top of script
- Wrap runnable code inside main() under if __name__ == "__main__":
- Parse all CLI flags with argparse; fail on unknown arguments
- Seed every RNG with 42 / random_state=42
- After finishing work, print the single line:  DONE {{json_blob}}  then STOP
- End every agent response with the token  {SENTINEL}
"""

# ---------------------------------------------------------------------
# 1. EDA AGENT
# ---------------------------------------------------------------------

eda_description = """
Role: Senior Data-Profiling Analyst
Mission: profile the train, validation, and test splits and write basic quality
reports for each, without extra hashing or size checks.
"""

eda_prompt = f"""
{GLOBAL_RULES}

### INPUTS
- TRAIN = '{{train_csv}}'
- VAL   = '{{val_csv}}'
- TEST  = '{{test_csv}}'

### TASK
1. Return **eda.py** inside ```python``` fences.
   - Accept CLI flags  --train_csv  --val_csv  --test_csv
   - For each split create:
       ART_DIR/train_info.txt          (df.info())
       ART_DIR/train_describe.csv      (df.describe(include='all').T)
       ART_DIR/train_missing.csv       (column, missing_pct)
       … same for val_* and test_* …
   - Print exactly:
     DONE {{"artefact_paths": "<semicolon-separated list>"}}
     then STOP
2. Emit a ```bash``` block that:
   - chmod +x eda.py
   - runs:  
     python eda.py \\
        --train_csv "{{train_csv}}" \\
        --val_csv   "{{val_csv}}"   \\
        --test_csv  "{{test_csv}}"
3. End with {SENTINEL}
{SENTINEL}
"""

# ---------------------------------------------------------------------
# 2. FEATURE-ENGINEERING AGENT
# ---------------------------------------------------------------------

feature_description = """
Role: Feature-Engineering Architect
Mission: convert raw data into model-ready features with robust, version-pinned
preprocessing.
"""

feature_prompt = f"""
{GLOBAL_RULES}

### INPUTS
- ART_DIR/info.txt, describe.csv, missing.csv
- Raw CSV at {{file}}

### TASK
1. Return feature.py in ```python``` fences.
   - Start with DEPENDENCY_CHECK
   - Define make_features(raw_path: str) -> pandas.DataFrame
     * impute numeric NaN with median; categoricals with mode
     * one-hot encode categoricals ≤20 unique
     * standard-scale numeric columns
   - Save ART_DIR/features.parquet (snappy)
   - When executed, parse --data_path and print
     DONE {{"out_path": "artifacts/features.parquet"}}
     then STOP
2. Emit a ```bash``` block that chmods and runs feature.py with --data_path "{{file}}"
3. End with {SENTINEL}
{SENTINEL}
"""

# ---------------------------------------------------------------------
# 3. MODELLING AGENT
# ---------------------------------------------------------------------

model_description = """
Role: AutoML Optimisation Scientist
Mission: sweep ensemble and gradient-boosted models with Optuna, retrain the
winner, and persist a reproducible model asset.
"""

model_prompt = f"""
{GLOBAL_RULES}

### TASK
1. Return model.py inside ```python``` fences.
   - Starts with DEPENDENCY_CHECK
   - Loads ART_DIR/features.parquet (expects column "target")
   - Splits 80-20 (random_state=42)
   - Uses optuna.create_study(direction="minimize") with 50 trials, 300-s timeout
     to tune:
     GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor,
     LGBMRegressor, XGBRegressor, CatBoostRegressor
   - Retrains best model on full data, saves to ART_DIR/best_model.joblib
   - Prints
     DONE {{"best_rmse": <float4>, "best_algo": "<name>"}}
     then STOP
2. Emit a ```bash``` block that chmods and runs model.py
3. End with {SENTINEL}
{SENTINEL}
"""

# ---------------------------------------------------------------------
# 4. EVALUATION AGENT
# ---------------------------------------------------------------------

eval_description = """
Role: Reliability Metrics Auditor
Mission: load the persisted model, compute canonical regression metrics, write
them to disk, and output a DONE status JSON.
"""

eval_prompt = f"""
{GLOBAL_RULES}

### TASK
1. Return evaluate.py in ```python``` fences.
   - Starts with DEPENDENCY_CHECK
   - Loads ART_DIR/best_model.joblib and ART_DIR/features.parquet
   - Computes RMSE, MAE, R2
   - Saves ART_DIR/metrics.json with scores and UTC timestamp
   - Prints
     DONE {{"metrics_path": "artifacts/metrics.json"}}
     then STOP
2. Emit a ```bash``` block that chmods and runs evaluate.py
3. End with {SENTINEL}
{SENTINEL}
"""