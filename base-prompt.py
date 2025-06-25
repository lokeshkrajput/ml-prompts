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
- Use only: stdlib, numpy, pandas, scikit-learn, lightgbm, xgboost, catboost, joblib, optuna, matplotlib
- Begin every script with the DEPENDENCY_CHECK block exactly as shown
- Wrap runnable code inside main() under if __name__ == "__main__":
- Parse all CLI flags with argparse; fail on unknown arguments
- Seed every RNG with 42
- Print DONE followed by a one-line JSON object, then STOP
- Finish each response with the token  {SENTINEL}
"""


eda_description = (
    "Role: Senior Data-Profiling Analyst.  \n"
    "Mission: generate a portable EDA script that profiles the raw CSV, "
    "writes schema and quality artefacts, and emits a machine-readable DONE "
    "status for downstream agents."
)

eda_prompt = f"""
{GLOBAL_RULES}

### TASK
1. Return eda.py inside ```python``` fences.
   - Script must accept --data_path
   - Produce artifacts/info.txt, artifacts/describe.csv, artifacts/missing.csv
   - Print exactly
     DONE {{"artefact_paths": "<semicolon-separated list>"}}
   - Stop after the DONE line
2. Emit a ```bash``` block that
   - chmod +x eda.py
   - runs eda.py with --data_path "{{file}}"
3. End with {SENTINEL}.
{SENTINEL}
"""


feature_description = (
    "Role: Feature-Engineering Architect.  \n"
    "Mission: convert raw data into model-ready features with robust, "
    "version-pinned preprocessing while writing a deterministic DONE status."
)

feature_prompt = f"""
{GLOBAL_RULES}

### INPUTS
- artifacts/info.txt, describe.csv, missing.csv
- Raw CSV at {{file}}

### TASK
1. Return feature.py in ```python``` fences.
   - Start with DEPENDENCY_CHECK
   - Define make_features(raw_path: str) -> pandas.DataFrame
     * Impute numeric NaN with median; categoricals with mode
     * One-hot encode categoricals with <= 20 unique values
     * Standard-scale numeric columns
   - Save artifacts/features.parquet (snappy)
   - When executed, parse --data_path and print
     DONE {{"out_path": "artifacts/features.parquet"}}
     then stop
2. Emit a ```bash``` block that chmods and runs feature.py with --data_path "{{file}}"
3. End with {SENTINEL}.
{SENTINEL}
"""

model_description = (
    "Role: AutoML Optimisation Scientist.  \n"
    "Mission: sweep ensemble/GBM algorithms with Optuna, retrain the winner, "
    "and persist a reproducible model asset."
)

model_prompt = f"""
{GLOBAL_RULES}

### TASK
1. Return model.py inside ```python``` fences.
   - Starts with DEPENDENCY_CHECK
   - Loads artifacts/features.parquet (expects column "target")
   - Splits 80-20 (random_state=42)
   - Uses Optuna (50 trials, 300 s timeout) to tune
     GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor,
     LGBMRegressor, XGBRegressor, CatBoostRegressor
   - Retrains best model on full data, saves to artifacts/best_model.joblib
   - Prints
     DONE {{"best_rmse": <float4>, "best_algo": "<name>"}}
     then stops
2. Emit a ```bash``` block that chmods and runs model.py
3. End with {SENTINEL}.
{SENTINEL}
"""


eval_description = (
    "Role: Reliability Metrics Auditor.  \n"
    "Mission: load the persisted model, compute canonical regression metrics, "
    "record them to disk, and output a DONE status JSON."
)

eval_prompt = f"""
{GLOBAL_RULES}

### TASK
1. Return evaluate.py in ```python``` fences.
   - Starts with DEPENDENCY_CHECK
   - Loads artifacts/best_model.joblib and artifacts/features.parquet
   - Computes RMSE, MAE, R2
   - Saves artifacts/metrics.json with scores and UTC timestamp
   - Prints
     DONE {{"metrics_path": "artifacts/metrics.json"}}
     then stops
2. Emit a ```bash``` block that chmods and runs evaluate.py
3. End with {SENTINEL}.
{SENTINEL}
"""