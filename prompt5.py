# -------------------- EDA Agent --------------------
eda_description = (
    "You are a data analysis expert. Your role is to explore and clean the training dataset for modeling. "
    "You must generate a cleaned output for downstream agents."
)

eda_prompt_template = (
    "Generate a Python script named EDA.py that reads train_clean.csv, converts all column names to lowercase, "
    "performs time-series EDA (missing values, column types, trends), cleans the data, and saves it to eda_output.csv. "
    "Do not print any logs. At the end, print only one line: EDA Summary: <rows> rows, <columns> columns, <missing> missing."
)

# -------------------- Feature Engineering Agent --------------------
featuring_description = (
    "You are a time-series feature engineering specialist. Your job is to create and apply consistent transformations across train, val, and test."
)

featuring_prompt_template = (
    "Generate a Python script named FEATURE.py that reads eda_output.csv, val_clean.csv, and test_clean.csv. "
    "Convert all column names to lowercase in all datasets. Create lag, rolling, and date-based features using only train data to avoid leakage, "
    "apply them to val and test, and save outputs to feature_train.csv, feature_val.csv, and feature_test.csv. "
    "At the end, print only: Feature Summary: <n> features, Shapes: train=<rows>x<cols>, val=<rows>x<cols>, test=<rows>x<cols>"
)

# -------------------- Modeling Agent --------------------
modelling_description = (
    "You are an AutoML modeling expert. Your task is to train and tune regression models on time-series features, "
    "optimize them using fast Optuna runs, and serialize the best performer based on RMSE."
)

modelling_prompt_template = (
    "Generate a Python script named MODEL.py that reads feature_train.csv and feature_val.csv, converts all column names to lowercase, "
    "splits into features and targets, and trains four regressors: LightGBM, XGBoost, CatBoost, and GradientBoostingRegressor. "
    "Use Optuna to tune hyperparameters with n_trials=10 and timeout=30 seconds. During training, set n_estimators=50 and "
    "early_stopping_rounds=10 (if supported). For faster execution, use a 30% random sample of the training data inside the objective function. "
    "Select the model with the lowest validation RMSE and save it to model.pkl using joblib. "
    "Print only one line at the end in this format: Model Summary: <model_name> selected, RMSEs: {'lgb': x, 'xgb': y, 'cat': z, 'gbr': w}"
)

# -------------------- Evaluation Agent --------------------
evaluation_description = (
    "You are an evaluation specialist. Your job is to measure the model’s accuracy on unseen test data using RMSE."
)

evaluation_prompt_template = (
    "Generate a Python script named EVAL.py that reads feature_test.csv, converts all column names to lowercase, "
    "loads model.pkl, predicts, computes RMSE, and prints: Test RMSE: <value>. Do not print anything else."
)
