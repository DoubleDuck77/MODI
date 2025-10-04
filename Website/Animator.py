import pandas as pd

# Load the CSV file
file_path = "~/MODI/KOI_lightcurves_fin/757450.csv"
df = pd.read_csv(file_path)

# Show basic info: columns and first few rows
df_info = {
    "columns": df.columns.tolist(),
    "head": df.head(10).to_dict(orient="records"),
    "num_rows": len(df)
}

df_info
