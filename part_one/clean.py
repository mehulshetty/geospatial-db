### clean.py --- Cleans the Dataset
import pandas as pd 

df = pd.read_csv('../dataset/nyc_dataset.csv')

# Remove Empty Entries
df.dropna(subset=['@id', '@lat', '@lon'], inplace=True)

# Check if IDs are all numeric
df['@id'] = df['@id'].astype(str)
df = df[df['@id'].str.isdigit()]

# Drop Duplicates
df.drop_duplicates(subset=['@id'], inplace=True)

# Save Cleaned Dataset
df.to_csv('../dataset/clean_nyc_dataset.csv')