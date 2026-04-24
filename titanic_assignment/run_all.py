from scripts.data_cleaning import clean_data
from scripts.feature_engineering import engineer_features
from scripts.feature_selection import select_features

print("Step 1: Cleaning Data...")
cleaned_path = clean_data("data/train.csv")

print("Step 2: Engineering Features...")
engineered_path = engineer_features(cleaned_path)

print("Step 3: Selecting Features...")
select_features(engineered_path)

print("\nAssignment Workflow Finished Successfully!")