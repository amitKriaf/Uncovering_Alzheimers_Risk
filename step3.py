#selects N participants from control group randomly 


import pandas as pd

# Parameters
controls_input = "/home/tzlilse@mta.ac.il/alzheimer_controls.csv"
sample_output = "/home/tzlilse@mta.ac.il/random_controls_sample.csv"
n = 7292  # Number of participants to randomly select

# Load control group
print("Loading control group...")
df_controls = pd.read_csv(controls_input)

# Filter only participants older than 70
df_controls_over_70 = df_controls[df_controls['age'] > 70]

# Ensure there are enough participants after filtering
if len(df_controls_over_70) < n:
    raise ValueError(f"Not enough participants over age 70 (found {len(df_controls_over_70)}, need {n})")

# Sample n participants randomly (without replacement)
print(f"Sampling {n} random participants over age 70 from control group...")
sample_df = df_controls_over_70.sample(n=n, random_state=42)

# Save to new file
sample_df.to_csv(sample_output, index=False)

# Report
print(f"Random control sample saved to: {sample_output} ({len(sample_df)} rows)")
