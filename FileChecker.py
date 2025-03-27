import pandas as pd

# Load the .parquet file
file_path = "logos.snappy.parquet"
try:
    df = pd.read_parquet(file_path, engine="pyarrow")  # Use the "pyarrow" or "fastparquet" engine
    print(df.info())  # Display data structure and basic details
    print(df.head())  # Preview the first few rows
except Exception as e:
    print(f"Error reading the .parquet file: {e}")