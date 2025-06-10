# filter relevant columns from biobank file

import pandas as pd

input_csv_path = "/home/tzlilse@mta.ac.il/biobank/ukb672220.csv"
output_csv_path = "/home/tzlilse@mta.ac.il/filtered_biobank_data.csv"
fields_output_local = "full_fields_name.txt"

columns_to_display = [
    "131036-0.0", "eid", "31-0.0", "34-0.0", "6138-0.0", "21000-0.0", "26417-0.0",
    "20116-0.0", "20160-0.0", "130836-0.0", "3894-0.0", "42020-0.0",
    "1558-0.0", "884-0.0", "1080-0.0", "1070-0.0", "874-0.0", "914-0.0",
    "894-0.0", "2149-0.0", "1538-0.0", "1498-0.0", "6144-0.0", "1548-0.0",
    "1050-0.0", "1717-0.0", "2267-0.0", "1160-0.0", "1180-0.0",
    "1220-0.0", "1190-0.0", "1110-0.0", "20240-0.0", "20191-0.0", "20132-0.0", "709-0.0",
    "53-0.0"
]

print("Reading header only...")
df_head = pd.read_csv(input_csv_path, nrows=0)
with open(fields_output_local, "w") as f:
    for col in df_head.columns.tolist():
        f.write(col + "\n")
print("Column names saved to file.")

print("Starting chunked reading and filtering...")
reader = pd.read_csv(input_csv_path, usecols=columns_to_display, chunksize=100000)

first_chunk = True
for i, chunk in enumerate(reader):
    print(f"Saving chunk {i + 1}, rows: {len(chunk)}")
    mode = 'w' if first_chunk else 'a'
    chunk.to_csv(output_csv_path, index=False, header=first_chunk, mode=mode)
    first_chunk = False

print(f"Done. Filtered file saved to: {output_csv_path}")