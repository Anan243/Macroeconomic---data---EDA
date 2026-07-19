"""
==============================================================

02     feature_engineering.py

Global Macroeconomic Structural Analysis

Purpose
-------
This script creates additional macroeconomic indicators
from the cleaned dataset.

The indicators created here will later be used for

• Univariate Analysis
• Bivariate Analysis
• Time Series Analysis
• Country Comparison
• Continent Comparison
• Power BI Dashboard

==============================================================
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

folders = [
    "Output",
    "Output/Feature Engineering",
    "Graphs"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    r"Output/latest_country_data.csv"
)

print("\nDataset Loaded Successfully")
print(df.shape)

# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
      .str.replace("(", "", regex=False)
      .str.replace(")", "", regex=False)
      .str.replace("-", "_")
      .str.replace(",", "")
)

print("\nColumn Names\n")
for c in df.columns:
    print(c)

# ============================================================
# FILTER FOR YEAR 2021 AND DROP OTHERS
# ============================================================

if "year" in df.columns:
    initial_shape = df.shape[0]
    df = df[df["year"] == 2021].copy()
    print(f"\nFiltered for Year = 2021. Rows reduced from {initial_shape} to {df.shape[0]}.")
else:
    print("\nWARNING: 'year' column not found. Skipping 2021 filtration.")

# ============================================================
# HELPER FUNCTION
#
# Finds a column automatically by keyword.
#
# This avoids KeyErrors if the dataset changes.
# ============================================================

def find_column(keyword):
    keyword = keyword.lower()
    for column in df.columns:
        if keyword in column:
            return column
    return None

# ============================================================
# FIND IMPORTANT VARIABLES
# ============================================================

gdp_col = find_column("gross_domestic_product")
exports_col = find_column("exports")
imports_col = find_column("imports")
investment_col = find_column("gross_capital_formation")
population_col = find_column("population")
gni_col = find_column("per_capita_gni")
government_col = find_column("general_government_final_consumption")
household_col = find_column("household_consumption")
manufacturing_col = find_column("manufacturing_isic_d")
agriculture_col = find_column("agriculture")
services_col = find_column("other_activities")

print("\nDetected Columns")
print("---------------------------")

important = {
    "GDP": gdp_col,
    "Exports": exports_col,
    "Imports": imports_col,
    "Investment": investment_col,
    "Population": population_col,
    "Per Capita GNI": gni_col,
    "Government": government_col,
    "Household": household_col,
    "Manufacturing": manufacturing_col,
    "Agriculture": agriculture_col,
    "Services": services_col
}

for k, v in important.items():
    print(f"{k:20} -> {v}")

# ============================================================
# ENSURE NUMERIC DATA
# ============================================================

numeric_columns = [
    gdp_col, exports_col, imports_col, investment_col,
    population_col, gni_col, government_col, household_col,
    manufacturing_col, agriculture_col, services_col
]

for col in numeric_columns:
    if col is not None:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nNumeric Conversion Complete")

# ============================================================
# REMOVE IMPOSSIBLE VALUES
# ============================================================

positive_columns = [
    gdp_col, exports_col, imports_col, investment_col,
    population_col, gni_col
]

for col in positive_columns:
    if col is not None:
        df.loc[df[col] < 0, col] = np.nan

print("\nImpossible values cleaned.")

# ============================================================
# PART 2 - CREATE MACROECONOMIC FEATURES
# ============================================================

print("\n")
print("="*60)
print("CREATING MACROECONOMIC FEATURES")
print("="*60)

if exports_col and gdp_col:
    df["export_dependency"] = df[exports_col] / df[gdp_col]

if imports_col and gdp_col:
    df["import_dependency"] = df[imports_col] / df[gdp_col]

if exports_col and imports_col and gdp_col:
    df["trade_openness"] = (df[exports_col] + df[imports_col]) / df[gdp_col]

if investment_col and gdp_col:
    df["investment_ratio"] = df[investment_col] / df[gdp_col]

if government_col and gdp_col:
    df["government_share"] = df[government_col] / df[gdp_col]

if household_col and gdp_col:
    df["household_share"] = df[household_col] / df[gdp_col]

if manufacturing_col and gdp_col:
    df["manufacturing_share"] = df[manufacturing_col] / df[gdp_col]

if agriculture_col and gdp_col:
    df["agriculture_share"] = df[agriculture_col] / df[gdp_col]

if services_col and gdp_col:
    df["services_share"] = df[services_col] / df[gdp_col]

if gdp_col and population_col:
    df["gdp_per_person"] = df[gdp_col] / df[population_col]

if investment_col and population_col:
    df["investment_per_person"] = df[investment_col] / df[population_col]

if exports_col and population_col:
    df["exports_per_person"] = df[exports_col] / df[population_col]

if imports_col and population_col:
    df["imports_per_person"] = df[imports_col] / df[population_col]

print("\nMacroeconomic Ratios Created Successfully")

# ============================================================
# CREATE QUARTILE GROUPS
# ============================================================

def create_quartile_group(column, new_column):
    if column not in df.columns:
        return

    temp = df[column].dropna()
    if temp.nunique() < 4:
        print(f"Skipped {new_column} (not enough unique values)")
        return

    df[new_column] = pd.qcut(
        df[column],
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
        duplicates="drop"
    )

print("\nCreating Quartile Groups...")
create_quartile_group(gdp_col, "gdp_group")
create_quartile_group(population_col, "population_group")
create_quartile_group(gni_col, "gni_group")
create_quartile_group("trade_openness", "trade_group")
create_quartile_group("investment_ratio", "investment_group")
create_quartile_group("export_dependency", "export_group")
create_quartile_group("import_dependency", "import_group")
create_quartile_group("government_share", "government_group")
create_quartile_group("household_share", "consumption_group")
create_quartile_group("manufacturing_share", "manufacturing_group")
create_quartile_group("agriculture_share", "agriculture_group")
create_quartile_group("services_share", "services_group")

print("\nQuartile Groups Created Successfully")

# ============================================================
# DISPLAY NEW FEATURES
# ============================================================

print("\nNew Feature Columns:\n")
engineered_features = [
    "export_dependency", "import_dependency", "trade_openness",
    "investment_ratio", "government_share", "household_share",
    "manufacturing_share", "agriculture_share", "services_share",
    "gdp_per_person", "investment_per_person", "exports_per_person",
    "imports_per_person"
]

for feature in engineered_features:
    if feature in df.columns:
        print(feature)

# ============================================================
# PART 3 - VALIDATION, SUMMARY TABLES AND EXPORTS
# ============================================================

print("\n")
print("="*70)
print("VALIDATING ENGINEERED FEATURES")
print("="*70)

existing_features = [col for col in engineered_features if col in df.columns]

# Initialize baseline summary statistics framework
feature_summary = df[existing_features].describe().T

# Adding diagnostics metrics
feature_summary["missing_values"] = df[existing_features].isnull().sum()
feature_summary["missing_percent"] = (df[existing_features].isnull().mean() * 100).round(2)

# New functionality: Extract country names mapping to min/max thresholds
min_countries = []
max_countries = []

for feature in existing_features:
    if "country" in df.columns and df[feature].notnull().any():
        # Identify positional indices for minimum and maximum entries
        min_idx = df[feature].idxmin()
        max_idx = df[feature].idxmax()
        
        # Pull related textual values from country row mapping
        min_countries.append(df.loc[min_idx, "country"])
        max_countries.append(df.loc[max_idx, "country"])
    else:
        min_countries.append(np.nan)
        max_countries.append(np.nan)

feature_summary["min_country"] = min_countries
feature_summary["max_country"] = max_countries

# Re-ordering summary frame structure to keep diagnostic metrics cleanly together
ordered_cols = [
    "count", "mean", "std", "min", "min_country", "25%", "50%", "75%", 
    "max", "max_country", "missing_values", "missing_percent"
]
feature_summary = feature_summary[[c for c in ordered_cols if c in feature_summary.columns]]

# Export updated diagnostics tracking profile
feature_summary.to_csv("Output/Feature Engineering/feature_summary_statistics.csv")
print("\nFeature Summary Created with Country Extremes.")

# ============================================================
# COUNTRY RANKINGS
# ============================================================

ranking_variables = [
    gdp_col, gni_col, "trade_openness", "investment_ratio",
    "export_dependency", "manufacturing_share", "services_share"
]

for variable in ranking_variables:
    if variable is None or variable not in df.columns:
        continue

    ranking = df[["country", variable]].sort_values(by=variable, ascending=False)
    filename = f"Output/Feature Engineering/{variable}_ranking.csv"
    ranking.to_csv(filename, index=False)

print("Country rankings exported.")

# ============================================================
# CONTINENT SUMMARY
# ============================================================

if "continent" in df.columns:
    continent_summary = df.groupby("continent")[existing_features].mean().round(3)
    continent_summary.to_csv("Output/Feature Engineering/continent_summary.csv")
    print("Continent Summary Created.")

# ============================================================
# DEVELOPMENT STATUS SUMMARY
# ============================================================

if "development_status" in df.columns:
    # 1. Compute the mathematical averages for numeric features
    development_summary = (
        df
        .groupby("development_status")[existing_features]
        .mean()
        .round(3)
    )
    
    # 2. Extract and concatenate the unique country strings for each development group
    if "country" in df.columns:
        country_lists = (
            df.dropna(subset=["development_status"])
            .groupby("development_status")["country"]
            .apply(lambda countries: ", ".join(sorted(countries.unique())))
        )
        
        # 3. Inject the joined strings into a new 'countries' column at the front
        development_summary.insert(0, "countries", country_lists)

    # Export the descriptive cluster profiles
    development_summary.to_csv(
        "Output/Feature Engineering/development_summary.csv"
    )

    print("Development Summary Created with country mappings.")




# ============================================================
# INCOME GROUP SUMMARY
# ============================================================

if "income_group" in df.columns:
    # 1. Compute the mathematical averages for numeric features
    income_summary = df.groupby("income_group")[existing_features].mean().round(3)
    
    # 2. Extract and concatenate the unique country strings for each bracket
    if "country" in df.columns:
        country_lists = (
            df.dropna(subset=["income_group"])
            .groupby("income_group")["country"]
            .apply(lambda countries: ", ".join(sorted(countries.unique())))
        )
        
        # 3. Inject the joined strings into a new 'countries' column at the front
        income_summary.insert(0, "countries", country_lists)
    
    # Export the descriptive cluster profiles
    income_summary.to_csv("Output/Feature Engineering/income_summary.csv")
    print("Income Group Summary Created with country mappings.")

# ============================================================
# VALIDATION CHECKS
# ============================================================

print("\n")
print("="*70)
print("VALIDATION CHECKS")
print("="*70)

for feature in existing_features:
    print("\n", feature)
    print("-------------------------")
    print("Minimum :", df[feature].min())
    print("Maximum :", df[feature].max())
    print("Mean    :", round(df[feature].mean(), 4))

# ============================================================
# TOP 10 COUNTRIES
# ============================================================

if gdp_col:
    print("\n" + "="*70 + "\nTOP 10 GDP COUNTRIES\n" + "="*70)
    print(df[["country", gdp_col]].sort_values(by=gdp_col, ascending=False).head(10))

if "trade_openness" in df.columns:
    print("\n" + "="*70 + "\nTOP 10 TRADE OPEN ECONOMIES\n" + "="*70)
    print(df[["country", "trade_openness"]].sort_values(by="trade_openness", ascending=False).head(10))

# ============================================================
# SAVE FINAL FEATURE ENGINEERED DATASET
# ============================================================

output_file = "Output/Feature Engineering/feature_engineeredd_dataset.csv"
df.to_csv(output_file, index=False)

print("\n" + "="*70 + "\nFEATURE ENGINEERING COMPLETE\n" + "="*70)
print(f"\nSaved:\n{output_file}\n\nSummary Tables\nRanking Tables\nFeature Statistics\nValidation Results\n\nReady for Exploratory Data Analysis.")
