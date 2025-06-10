# split to "alzheimer_controls" and "alzheimer_cases"

import pandas as pd

# Paths
filtered_input = "/home/tzlilse@mta.ac.il/filtered_biobank_data_new.csv"
cases_output = "/home/tzlilse@mta.ac.il/alzheimer_cases.csv"
controls_output = "/home/tzlilse@mta.ac.il/alzheimer_controls.csv"
target_column = "has alzheimer"

# Read filtered data
print("Loading filtered dataset...")
df = pd.read_csv(filtered_input)

# Split by Alzheimer diagnosis column
print("Splitting into Alzheimer cases and controls...")
cases_df = df[df[target_column]==1]
controls_df = df[df[target_column]==0]

# Count participants
num_cases = len(cases_df)
num_controls = len(controls_df)

# Save to separate CSVs
cases_df.to_csv(cases_output, index=False)
controls_df.to_csv(controls_output, index=False)

# Report
print(f"Saved Alzheimer cases to: {cases_output} ({num_cases} participants)")
print(f"Saved control group to: {controls_output} ({num_controls} participants)")
print("Split completed successfully.")