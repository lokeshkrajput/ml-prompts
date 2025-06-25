
SENTINEL = "###END_OF_AGENT_OUTPUT###"
ART_DIR  = "artifacts"

# Shared rules
GLOBAL_RULES = """
- Begin every generated script with the DEPENDENCY_CHECK block (see repo docs)
- Define ART_DIR = "artifacts" once at top
- Wrap runnable code inside main() under if __name__ == "__main__":
- Parse CLI flags with argparse; fail on unknown flags
- Seed numpy / random / LightGBM with 42
- After finishing, print a single line:  DONE {"status": "ok"}  then STOP
- End each agent response with the sentinel token  ###END_OF_AGENT_OUTPUT###
"""

eda_description = """
Role: Financial EDA Specialist
Mission: Validate the schema (Date, Open, High, Low, Close, Volume), cast dtypes,
flag gaps or timezone issues, deduplicate duplicate Date rows by averaging,
sort chronologically, then write version-tagged train_clean, val_clean and
test_clean CSVs plus a minimal JSON profile for downstream integrity checks.
"""

eda_prompt_template = """{GLOBAL_RULES}

### INPUT FLAGS
--train_csv {{train_csv}}   --val_csv {{val_csv}}   --test_csv {{test_csv}}

### TASK (ordered)
1. Read each split; aggregate duplicate 'Date' rows using mean.
2. Convert 'Date' to pandas datetime, set as index and sort ascending.
3. Save cleaned splits to:
   - artifacts/train_clean.csv
   - artifacts/val_clean.csv
   - artifacts/test_clean.csv
4. Print the DONE line and stop.

Return the script within ```python``` fences, followed by a ```bash``` block
that executes:

python eda.py \
  --train_csv {{train_csv}} \
  --val_csv   {{val_csv}} \
  --test_csv  {{test_csv}}

###END_OF_AGENT_OUTPUT###
""".format(GLOBAL_RULES=GLOBAL_RULES)


fe_description = """
Role: Feature-Signal Architect
Mission: Generate leak-free lag, rolling and technical-indicator features;
apply consistent scaling; persist the fitted transformation pipeline (joblib);
output numeric train/val/test feature CSVs; and log the final feature list for
model transparency.
"""

fe_prompt_template = """{GLOBAL_RULES}

### INPUT FILES
- artifacts/train_clean.csv
- artifacts/val_clean.csv
- artifacts/test_clean.csv

### TASK (ordered)
1. Create lag features (1, 2, 3, 5-day) and rolling stats (mean, std, min, max)
   for 5, 10, 20-day windows.
2. Optionally add RSI, MACD and Bollinger Bands.
3. Create column 'target' = next-day log return of 'Close'.
4. Drop rows with NaN; ensure all columns are numeric.
5. Persist the preprocessing pipeline to artifacts/pipeline.joblib.
6. Save feature splits to:
   - artifacts/train_features.csv
   - artifacts/val_features.csv
   - artifacts/test_features.csv
7. Print DONE line and stop.

Return FEATURE.py plus a runner bash block.

###END_OF_AGENT_OUTPUT###
""".format(GLOBAL_RULES=GLOBAL_RULES)


model_description = """
Role: AutoML Optimisation Scientist
Mission: With reproducible seeds, run an Optuna hyper-parameter search
(50 trials) on LightGBM using strict time-series CV and early stopping,
retrain the best model on combined train+val, and save both model.pkl and
training metadata (params, feature list, CV scores, feature importances).
"""

model_prompt_template = """{GLOBAL_RULES}

### INPUT FILES
- artifacts/train_features.csv
- artifacts/val_features.csv
- artifacts/test_features.csv

### TASK
1. Ensure LightGBM >= 4.0 is installed; pip-install if missing.
2. Split train vs val strictly by time order.
3. Run optuna.create_study(direction="minimize") for 50 trials on LGBMRegressor
   with early-stopping callbacks.
4. Retrain the best params on train+val and save to artifacts/model.pkl.
5. Write metadata JSON to artifacts/model_meta.json.
6. Print DONE {{"mae": <float>, "rmse": <float>}} and stop.

Return MODEL.py plus a bash runner block.

###END_OF_AGENT_OUTPUT###
""".format(GLOBAL_RULES=GLOBAL_RULES)


eval_description = """
Role: Model-Reliability Auditor
Mission: Load the persisted model, score the hold-out test split, compute RMSE,
MAE and sign accuracy, write RMSE to MSFT_Score.txt, store metrics.json and a
residual plot in artifacts, and warn if RMSE exceeds validation RMSE by >20%.
"""

eval_prompt_template = """{GLOBAL_RULES}

### INPUT FILES
- artifacts/model.pkl
- artifacts/test_features.csv
- artifacts/model_meta.json

### TASK
1. Load model and test data; verify column 'target' exists.
2. Compute RMSE, MAE and sign accuracy.
3. Write RMSE to MSFT_Score.txt (single float).
4. Save full metrics to artifacts/metrics.json and a residual plot PNG.
5. Print DONE line and stop.

Return EVAL.py and a bash runner block.

###END_OF_AGENT_OUTPUT###
""".format(GLOBAL_RULES=GLOBAL_RULES)
