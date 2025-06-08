
import json
import pandas as pd
from pathlib import Path

# File paths
log_file = Path("submission_log.json")
result_file = Path("submission_result.json")
excel_file = Path("agent_prompt_scores.xlsx")

# Load submission logs for prompts
with open(log_file, "r") as f:
    submission_log = json.load(f)

# Extract agent prompts
agent_prompts = {}
for agent_key, agent_data in submission_log.items():
    if agent_key.endswith("_Agent") and "prompt" in agent_data:
        agent_prompts[agent_key] = agent_data["prompt"]

# Load submission results for scores
with open(result_file, "r") as f:
    submission_result = json.load(f)

# Extract scores
agent_scores = {}
for key, val in submission_result.items():
    if isinstance(val, float):  # only keep actual scores
        agent_scores[key] = val

# Combine prompts and scores
agents = sorted(set(agent_prompts.keys()) | set(agent_scores.keys()))
data = {
    "Prompt": [agent_prompts.get(agent, "") for agent in agents],
    "Score": [agent_scores.get(agent, "") for agent in agents]
}
df = pd.DataFrame(data, index=agents)
df.index.name = "Agent"

# Transpose the data
df_transposed = df.T

# Append to existing Excel without overwriting
if excel_file.exists():
    with pd.ExcelWriter(excel_file, mode='a', if_sheet_exists='new') as writer:
        df_transposed.to_excel(writer, sheet_name=f"Run_{len(pd.read_excel(excel_file, sheet_name=None)) + 1}")
else:
    with pd.ExcelWriter(excel_file, mode='w') as writer:
        df_transposed.to_excel(writer, sheet_name="Run_1")

print(f"Appended results to: {excel_file.name}")
