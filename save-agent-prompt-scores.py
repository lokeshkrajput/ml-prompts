
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# File paths
log_file = Path("submission_log.json")
result_file = Path("submission_result.json")
excel_file = Path("backup/agent_prompt_scores.xlsx")

# Exit if submission_result.json does not exist
if not result_file.exists():
    print("submission_result.json not found. Skipping entry.")
    exit()

# Use file modification timestamp as Run Sequence
mod_time = datetime.fromtimestamp(result_file.stat().st_mtime)
run_timestamp = mod_time.strftime("%Y-%m-%d %H:%M:%S")

# Load logs and results
with open(log_file, "r") as f:
    submission_log = json.load(f)

with open(result_file, "r") as f:
    submission_result = json.load(f)

# Extract prompts
eda_prompt = submission_log.get("EDA_Agent", {}).get("prompt", "")
fe_prompt = submission_log.get("FeatureEngineering_Agent", {}).get("prompt", "")
model_prompt = submission_log.get("Modeling_Agent", {}).get("prompt", "")
eval_prompt = submission_log.get("Evaluation_Agent", {}).get("prompt", "")

# Extract scores
eda_score = submission_result.get("EDA_Agent", "")
fe_score = submission_result.get("FeatureEngineering_Agent", "")
model_score = submission_result.get("Modeling_Agent", "")
eval_score = submission_result.get("Evaluation_Agent", "")
rmse_score = submission_result.get("Global_RMSE_Score", "")
final_score = submission_result.get("Aggregated_Final_Score", "")

# Define ordered data
row_data = {
    "Run Sequence": run_timestamp,
    "EDA Prompt": eda_prompt,
    "EDA Score": eda_score,
    "FeatureEngineering Prompt": fe_prompt,
    "FeatureEngineering Score": fe_score,
    "Modeling Prompt": model_prompt,
    "Modeling Score": model_score,
    "Evaluation Prompt": eval_prompt,
    "Evaluation Score": eval_score,
    "Global RMSE Score": rmse_score,
    "Aggregated Final Score": final_score
}
df_row = pd.DataFrame([row_data])

# Check for duplicates and append only if not present
if excel_file.exists():
    df_existing = pd.read_excel(excel_file)
    if run_timestamp in df_existing["Run Sequence"].astype(str).values:
        print(f"Entry for timestamp {run_timestamp} already exists. Skipping append.")
    else:
        df_combined = pd.concat([df_existing, df_row], ignore_index=True)
        df_combined.to_excel(excel_file, index=False)
        print(f"Appended entry for {run_timestamp} to {excel_file.name}")
else:
    df_row.to_excel(excel_file, index=False)
    print(f"Created new Excel file and added entry for {run_timestamp}")
