# merge_datasets.py
# Script to merge the raw datasets from Kaggle Crop Yield Prediction Dataset with more columns
# Inputs: pesticides.csv, rainfall.csv, temp.csv, yield.csv in ./data/
# Output: enhanced_yield_df.csv in ./data/

import pandas as pd

# ------------------------------
# Load Datasets
# ------------------------------
pest_df = pd.read_csv("./data/pesticides.csv")
rain_df = pd.read_csv("./data/rainfall.csv")
temp_df = pd.read_csv("./data/temp.csv")
yield_df_raw = pd.read_csv("./data/yield.csv")

print("Loaded Datasets Shapes:")
print(f"Pesticides: {pest_df.shape}")
print(f"Rainfall: {rain_df.shape}")
print(f"Temperature: {temp_df.shape}")
print(f"Yield Raw: {yield_df_raw.shape}")

# ------------------------------
# Inspect Elements in Yield Data
# ------------------------------
print("Unique Elements in Yield Raw:", yield_df_raw['Element'].unique())

# ------------------------------
# Pivot Yield Data
# ------------------------------
yield_pivot = yield_df_raw.pivot_table(
    index=['Domain_Code', 'Domain', 'area_Code', 'area', 'Element_Code', 'Element', 'item_Code', 'item', 'year_Code', 'year', 'Unit'],
    columns='Element',
    values='Value',
    aggfunc='first'
).reset_index()

# Normalize column names
yield_pivot.columns = yield_pivot.columns.str.strip().str.lower().str.replace(' ', '_')

# Rename for clarity
yield_pivot = yield_pivot.rename(columns={
    'yield': 'hg/ha_yield',
    'area_harvested': 'area_harvested_ha',
    'production': 'production_tonnes'
})

print("Yield Pivot Columns:", yield_pivot.columns.tolist())

# ------------------------------
# Pivot Pesticides Data
# ------------------------------
pest_pivot = pest_df.pivot_table(
    index=['area', 'year'],
    columns='item',
    values='Value',
    aggfunc='sum',
    fill_value=0
).reset_index()

# ------------------------------
# Merge All Datasets
# ------------------------------
df = yield_pivot.merge(pest_pivot.rename(columns={'area': 'area', 'year': 'year'}), on=['area', 'year'], how='left')
df = df.merge(rain_df.rename(columns={'area': 'area', 'year': 'year'}), on=['area', 'year'], how='left')
temp_df = temp_df.rename(columns={'country': 'area', 'year': 'year'})
df = df.merge(temp_df, on=['area', 'year'], how='left')
print(f"Merged Dataset Shape After Joins: {df.shape}")

# ------------------------------
# Handle Missing Values
# ------------------------------
df = df.sort_values(['area', 'item', 'year'])
group_cols = ['area', 'item']

# Fill time-series features
fill_cols = [col for col in df.columns if col not in ['hg/ha_yield', 'area_harvested_ha', 'production_tonnes', 'area', 'item', 'year', 'Domain_Code', 'Domain', 'area_Code', 'Element_Code', 'Element', 'item_Code', 'year_Code', 'Unit']]
df[fill_cols] = df.groupby(group_cols)[fill_cols].ffill().bfill()

# Impute medians for yield-related columns
if 'hg/ha_yield' in df.columns:
    df['hg/ha_yield'] = df['hg/ha_yield'].fillna(df.groupby(group_cols)['hg/ha_yield'].transform('median'))
else:
    print("Warning: 'hg/ha_yield' column not found.")

if 'area_harvested_ha' in df.columns:
    df['area_harvested_ha'] = df['area_harvested_ha'].fillna(df.groupby(group_cols)['area_harvested_ha'].transform('median'))
else:
    print("Warning: 'area_harvested_ha' column not found.")

if 'production_tonnes' in df.columns:
    df['production_tonnes'] = df['production_tonnes'].fillna(df.groupby(group_cols)['production_tonnes'].transform('median'))
else:
    print("Warning: 'production_tonnes' column not found.")

# Drop rows missing target
df = df.dropna(subset=['hg/ha_yield'])

# ------------------------------
# Save Final Dataset
# ------------------------------
print(f"Merged Dataset Shape: {df.shape}")
print("Columns:", df.columns.tolist())
print(df.head())

df.to_csv("./data/enhanced_yield_df.csv", index=False)
print("Merged dataset saved as ./data/enhanced_yield_df.csv")
