import pandas as pd
import numpy as np

# Load the dataset
input_path = "/home/tzlilse@mta.ac.il/filtered_biobank_data.csv"
df = pd.read_csv(input_path)

df["age"] = 2025-df["34-0.0"]

df["53-0.0"] = pd.to_datetime(df["53-0.0"])

df["join date"] = df["53-0.0"].dt.year - df["34-0.0"]

df["131036-0.0"] = pd.to_datetime(df["131036-0.0"])

df["age got sick"] = df["131036-0.0"].dt.year - df["34-0.0"]

df["time till got sick"] = df["age got sick"]-df["join date"]

df = df[(((df["age got sick"]>=70)|(df["age got sick"].isna()))
&((df["time till got sick"]>=5)|(df["time till got sick"].isna())))] 

cols_to_fix = ["1050-0.0","1070-0.0","1080-0.0","1498-0.0"]

df[cols_to_fix] = df[cols_to_fix].replace(-10,0.5)


df[df.select_dtypes(include='number')<0]=np.nan


df["has alzheimer"] = df["131036-0.0"].notna().astype(int)

df.drop(["eid","53-0.0","3894-0.0","20191-0.0","20240-0.0", "20132-0.0","131036-0.0","42020-0.0","130836-0.0", "join date","time till got sick", "age got sick","34-0.0"], axis=1,inplace=True)

# df.drop(["age"], axis=1,inplace=True)

# Save the cleaned dataset
df.to_csv("/home/tzlilse@mta.ac.il/filtered_biobank_data_new.csv", index=False, encoding='utf-8')
print("\n Cleaned dataset saved. Ready for work.")

