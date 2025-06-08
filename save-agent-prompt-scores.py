
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# File paths
log_file = Path("submission_log.json")
result_file = Path("submission_result.json")
excel_file = Path("agent_prompt_scores.xlsx")

# Load submission logs for prompts
with open(log_file, "r") as f:
    submission_log = json.load(f)

# Load submission results for scores
with open(result_file, "r") as f:
    submission_result = json.load(f)

# Extract specific prompts
eda_prompt = submission_log.get("EDA_Agent", {}).get("prompt", "")
fe_prompt = submission_log.get("FeatureEngineering_Agent", {}).get("prompt", "")
model_prompt = submission_log.get("Modeling_Agent", {}).get("prompt", "")
eval_prompt = submission_log.get("Evaluation_Agent", {}).get("prompt", "")

# Extract specific scores
eda_score = submission_result.get("EDA_Agent", "")
fe_score = submission_result.get("FeatureEngineering_Agent", "")
model_score = submission_result.get("Modeling_Agent", "")
eval_score = submission_result.get("Evaluation_Agent", "")
rmse_score = submission_result.get("Global_RMSE_Score", "")
final_score = submission_result.get("Aggregated_Final_Score", "")

# Define ordered columns
ordered_data = {
    "Run Sequence": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    "EDA Prompt": [eda_prompt],
    "EDA Score": [eda_score],
    "FeatureEngineering Prompt": [fe_prompt],
    "FeatureEngineering Score": [fe_score],
    "Modeling Prompt": [model_prompt],
    "Modeling Score": [model_score],
    "Evaluation Prompt": [eval_prompt],
    "Evaluation Score": [eval_score],
    "Global RMSE Score": [rmse_score],
    "Aggregated Final Score": [final_score]
}

# Convert to DataFrame
df_row = pd.DataFrame(ordered_data)

# Append or create Excel
if excel_file.exists():
    existing_df = pd.read_excel(excel_file)
    combined_df = pd.concat([existing_df, df_row], ignore_index=True)
    combined_df.to_excel(excel_file, index=False)
else:
    df_row.to_excel(excel_file, index=False)

print(f"Data written to: {excel_file.name}")
