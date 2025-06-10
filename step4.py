# Combines Alzheimer cases and control group into one file with a has_alzheimer column.

import pandas as pd

# File paths
cases_path = "/home/tzlilse@mta.ac.il/alzheimer_cases.csv"
controls_path = "/home/tzlilse@mta.ac.il/random_controls_sample.csv"
output_path = "/home/tzlilse@mta.ac.il/combined_alzheimer_dataset.csv"

# Load both datasets
print("Loading Alzheimer cases and control sample...")
cases_df = pd.read_csv(cases_path)
controls_df = pd.read_csv(controls_path)

# Combine
print("Combining datasets...")
combined_df = pd.concat([cases_df, controls_df], ignore_index=True)

# Save to file
combined_df.to_csv(output_path, index=False)
print(f"Combined dataset saved to: {output_path} ({len(combined_df)} rows)")
