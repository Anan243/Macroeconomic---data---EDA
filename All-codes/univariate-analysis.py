"""
===============================================================
06_univariate_analysis.py

Purpose
Examine individual macroeconomic variables to understand their 
distributions, shapes, and identify extreme outlier countries.

Topics
1. Descriptive Statistics (Central Tendency & Dispersion)
2. Distribution Shape (Skewness & Kurtosis)
3. Outlier Detection (Interquartile Range Method)
4. Visual Distributions (Histograms, KDEs, and Boxplots)
===============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for high-quality publication graphics
sns.set_theme(style="whitegrid")

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================
folders = [
    "Output/Univariate Analysis",
    "Graphs/Univariate Analysis",
    "Graphs/Univariate Analysis/Distributions",
    "Graphs/Univariate Analysis/Boxplots"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(
    r"C:\\Users\\BALAJI\\Downloads\\global-econ-analysis\\Output\\Feature Engineering\\feature_engineeredd_dataset.csv"
)
print(f"Dataset Loaded. Shape: {df.shape}")

# ============================================================
# IDENTIFY NUMERIC VARIABLES
# ============================================================
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

# Exclude non-macro indicators like IDs, years, or zip codes if they accidentally look numeric
exclude_cols = ["year", "id", "postal", "code"]
macro_numeric_cols = [
    col for col in numeric_columns 
    if not any(x in col.lower() for x in exclude_cols)
]

print(f"\nFound {len(macro_numeric_cols)} macroeconomic numeric variables for analysis.")

# ============================================================
# PART 1: STATISTICAL SUMMARY MATRIX
# ============================================================
print("\nCalculating summary metrics matrix...")

summary_stats = []

for col in macro_numeric_cols:
    col_data = df[col].dropna()
    if len(col_data) == 0:
        continue
        
    # Calculate key metrics
    mean_val = col_data.mean()
    median_val = col_data.median()
    std_val = col_data.std()
    min_val = col_data.min()
    max_val = col_data.max()
    skew_val = col_data.skew()
    kurt_val = col_data.kurt()
    
    # Percentiles
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1
    
    # Boundary definitions for outliers
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    # Count outliers
    outliers_count = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
    
    summary_stats.append({
        "Variable": col,
        "Observations": len(col_data),
        "Mean": round(mean_val, 3),
        "Median": round(median_val, 3),
        "Std Dev": round(std_val, 3),
        "Min": round(min_val, 3),
        "Max": round(max_val, 3),
        "Skewness": round(skew_val, 3),
        "Kurtosis": round(kurt_val, 3),
        "Lower Bound (1.5 IQR)": round(lower_bound, 3),
        "Upper Bound (1.5 IQR)": round(upper_bound, 3),
        "Outlier Count": outliers_count
    })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv("Output/Univariate Analysis/univariate_summary_matrix.csv", index=False)
print("Univariate statistical summary saved to: Output/Univariate Analysis/univariate_summary_matrix.csv")


# ============================================================
# PART 2: DETAILED OUTLIER PROFILE LOGGING
# ============================================================
print("\nProfiling extreme outlier records...")

outlier_records = []

for col in macro_numeric_cols:
    col_data = df[col].dropna()
    if len(col_data) == 0:
        continue
        
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    # Find rows containing outliers for this specific column
    outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    outlier_df = df[outliers_mask].copy()
    
    for _, row in outlier_df.iterrows():
        val = row[col]
        direction = "High" if val > upper_bound else "Low"
        
        outlier_records.append({
            "Variable": col,
            "Country": row.get("country", "Unknown"),
            "Year": row.get("year", "Unknown"),
            "Value": val,
            "Outlier Type": direction,
            "Threshold Breached": upper_bound if direction == "High" else lower_bound
        })

if outlier_records:
    outlier_summary_df = pd.DataFrame(outlier_records)
    outlier_summary_df.to_csv("Output/Univariate Analysis/detailed_outlier_profile.csv", index=False)
    print("Detailed country outlier profiles saved to: Output/Univariate Analysis/detailed_outlier_profile.csv")


# ============================================================
# PART 3: DISTRIBUTION PLOTS GENERATION
# ============================================================
print("\nGenerating distribution visuals...")

# We loop through key indicators or all columns to build plots dynamically
for col in macro_numeric_cols:
    temp_data = df[col].dropna()
    if len(temp_data) < 5:
        continue
        
    clean_title = col.replace("_", " ").title()
    
    # 1. Histogram & KDE Distribution plot
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(temp_data, kde=True, color="teal", alpha=0.6, ax=ax)
    ax.axvline(temp_data.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {round(temp_data.mean(), 2)}")
    ax.axvline(temp_data.median(), color="blue", linestyle="-", linewidth=1.5, label=f"Median: {round(temp_data.median(), 2)}")
    ax.set_title(f"Distribution Profiling: {clean_title}")
    ax.set_xlabel(clean_title)
    ax.set_ylabel("Frequency Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"Graphs/Univariate Analysis/Distributions/{col}_distribution.png", dpi=300)
    plt.close(fig)
    
    # 2. Outlier Identification Boxplot (Changed color to valid hex #FFB000)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(x=temp_data, color="#FFB000", orient="h", ax=ax, flierprops={"markerfacecolor": "crimson", "marker": "D"})
    ax.set_title(f"Outlier Boundary Map (1.5 IQR): {clean_title}")
    ax.set_xlabel(clean_title)
    fig.tight_layout()
    fig.savefig(f"Graphs/Univariate Analysis/Boxplots/{col}_boxplot.png", dpi=300)
    plt.close(fig)

print("\nUnivariate Analysis Pipeline Complete. All graphics and matrices exported successfully.")