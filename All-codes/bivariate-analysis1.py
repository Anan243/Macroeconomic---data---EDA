"""
===============================================================

07_bivariate_analysis.py

Purpose

Study relationships between macroeconomic indicators.

Macroeconomic Concepts

• Economic Growth
• Trade
• Investment
• Structural Transformation
• Development Economics

===============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Added for pairplot generation
import os

# ===============================================================
# CREATE OUTPUT FOLDERS
# ===============================================================

folders = [
    "Output/Bivariate Analysis",
    "Graphs/Bivariate Analysis",
    "Graphs/Bivariate Analysis/Scatterplots",
    "Graphs/Bivariate Analysis/Correlation",
    "Graphs/Bivariate Analysis/Pairplots"  # Added folder for the new pairplot grid
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ===============================================================
# LOAD DATA
# ===============================================================

df = pd.read_csv(
    r"C:\\Users\\BALAJI\\Downloads\\global-econ-analysis\\Output\\Feature Engineering\\feature_engineeredd_dataset.csv"
)

print(df.shape)

# ===============================================================
# IDENTIFY NUMERIC VARIABLES
# ===============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print("\nNumeric Variables")
for col in numeric_columns:
    print(col)

# ===============================================================
# CORRELATION MATRIX
#
# Pearson Correlation
#
# Measures linear relationship.
#
# Range
#
# -1  to  +1
#
# ===============================================================

correlation = (
    df[numeric_columns]
    .corr(
        method="pearson"
    )
)

correlation.to_csv(
    "Output/Bivariate Analysis/correlation_matrix.csv"
)

print("\nCorrelation Matrix Created")

# ===============================================================
# GDP RELATIONSHIPS
#
# Find variables most strongly correlated
# with GDP.
#
# ===============================================================

gdp_candidates = [
    col
    for col in numeric_columns
    if "gross_domestic_product" in col.lower()
]

if len(gdp_candidates) > 0:
    gdp_col = gdp_candidates[0]
    gdp_corr = (
        correlation[gdp_col]
        .sort_values(
            ascending=False
        )
    )
    gdp_corr.to_csv(
        "Output/Bivariate Analysis/gdp_correlations.csv"
    )
    print("\nGDP Correlations")
    print(gdp_corr)

# ===============================================================
# GNI RELATIONSHIPS
# ===============================================================

gni_candidates = [
    col
    for col in numeric_columns
    if "gross_national_income" in col.lower()
]

if len(gni_candidates) > 0:
    gni_col = gni_candidates[0]
    gni_corr = (
        correlation[gni_col]
        .sort_values(
            ascending=False
        )
    )
    gni_corr.to_csv(
        "Output/Bivariate Analysis/gni_correlations.csv"
    )

print("\nCorrelation Files Saved")

# ===============================================================
# PART 2
#
# SCATTER PLOTS + REGRESSION LINES
#
# Purpose
# -------
# Examine relationships between important macroeconomic
# indicators.
#
# Each graph answers a specific economic question.
#
# ===============================================================

# ===============================================================
# Helper Function
#
# Creates
# • Scatter Plot
# • Best Fit Regression Line
# • Correlation Value
# ===============================================================

def scatter_regression(x_column,
                       y_column,
                       title,
                       x_label,
                       y_label,
                       filename):

    if x_column not in df.columns:
        return

    if y_column not in df.columns:
        return

    temp = df[[x_column, y_column]].dropna()

    if len(temp) < 5:
        return

    x = temp[x_column]
    y = temp[y_column]

    plt.figure(figsize=(9,6))

    plt.scatter(
        x,
        y,
        alpha=0.6
    )

    # Regression Line
    m, b = np.polyfit(x, y, 1)
    plt.plot(
        x,
        m*x + b,
        linewidth=2
    )

    correlation = x.corr(y)
    plt.title(
        title +
        "\nCorrelation = "
        +
        str(round(correlation,3))
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "Graphs/Bivariate Analysis/Scatterplots/"
        + filename,
        dpi=300
    )
    plt.close()

# ===============================================================
# FIND IMPORTANT VARIABLES
# ===============================================================

def find_column(keyword):
    for column in df.columns:
        if keyword.lower() in column.lower():
            return column
    return None

gdp_col = find_column("gross_domestic_product")
gni_col = find_column("gross_national_income")
population_col = find_column("population")
exports_col = find_column("exports")
imports_col = find_column("imports")
investment_col = find_column("gross_capital_formation")

# ===============================================================
# GDP vs GNI
#
# Do countries with higher GDP also have
# higher Gross National Income?
#
# ===============================================================

scatter_regression(
    gdp_col,
    gni_col,
    "GDP vs GNI",
    "GDP",
    "GNI",
    "GDP_vs_GNI.png"
)

# ===============================================================
# GDP vs Population
#
# Larger countries generally produce more.
#
# ===============================================================

scatter_regression(
    population_col,
    gdp_col,
    "Population vs GDP",
    "Population",
    "GDP",
    "Population_vs_GDP.png"
)

# ===============================================================
# GDP vs Exports
#
# Export-led Growth Theory
#
# ===============================================================

scatter_regression(
    exports_col,
    gdp_col,
    "Exports vs GDP",
    "Exports",
    "GDP",
    "Exports_vs_GDP.png"
)

# ===============================================================
# GDP vs Imports
#
# Imports often rise with economic activity.
#
# ===============================================================

scatter_regression(
    imports_col,
    gdp_col,
    "Imports vs GDP",
    "Imports",
    "GDP",
    "Imports_vs_GDP.png"
)

# ===============================================================
# GDP vs Investment
#
# Solow Growth Model
#
# Higher investment usually increases
# productive capacity.
#
# ===============================================================

scatter_regression(
    investment_col,
    gdp_col,
    "Investment vs GDP",
    "Investment",
    "GDP",
    "Investment_vs_GDP.png"
)

# ===============================================================
# TRADE OPENNESS vs GDP
#
# ===============================================================

if "trade_openness" in df.columns:
    scatter_regression(
        "trade_openness",
        gdp_col,
        "Trade Openness vs GDP",
        "Trade Openness",
        "GDP",
        "TradeOpenness_vs_GDP.png"
    )

# ===============================================================
# INVESTMENT RATIO vs GDP GROWTH
#
# ===============================================================

if "investment_ratio" in df.columns and "gdp_growth_rate" in df.columns:
    scatter_regression(
        "investment_ratio",
        "gdp_growth_rate",
        "Investment Ratio vs GDP Growth",
        "Investment Ratio",
        "GDP Growth",
        "InvestmentRatio_vs_GDPGrowth.png"
    )

print("\nScatterplots Created Successfully.")


# ===============================================================
# PAIRWISE SCATTERPLOT MATRIX (PAIRPLOT)
#
# Renders a complete subdivisions matrix plot showing how key 
# macroeconomic columns relate to each other in a single canvas.
# ===============================================================

print("\nGenerating Pairwise Matrix Chart (Pairplot)...")

# Select key representative variables (too many variables makes the subplots unreadable)
pairplot_vars = [
    col for col in [
        "trade_openness", 
        "investment_ratio", 
        "gdp_per_person", 
        "manufacturing_share", 
        "services_share"
    ] if col in df.columns
]

if len(pairplot_vars) >= 2:
    # Set the aesthetics
    sns.set_theme(style="ticks")
    
    # We include 'income_group' or 'development_status' as hue if present
    hue_col = None
    if "income_group" in df.columns and df["income_group"].nunique() > 1:
        hue_col = "income_group"
    elif "development_status" in df.columns and df["development_status"].nunique() > 1:
        hue_col = "development_status"
        
    pair_grid = sns.pairplot(
        df if hue_col else df[pairplot_vars].dropna(),
        vars=pairplot_vars,
        hue=hue_col,
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "edgecolor": "none"},
        corner=False  # Keep True if you only want the bottom-left triangle
    )
    
    pair_grid.fig.suptitle("Pairwise Macroeconomic Indicator Relationships Matrix", y=1.02, fontsize=16)
    
    pairplot_path = "Graphs/Bivariate Analysis/Pairplots/macroeconomic_indicators_pairplot.png"
    pair_grid.savefig(pairplot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Pairplot Matrix exported successfully to: {pairplot_path}")
else:
    print("WARNING: Skipping pairplot. Not enough key columns found.")


# ===============================================================
# PART 3
#
# CORRELATION HEATMAPS
# SPEARMAN CORRELATION
# STRONGEST RELATIONSHIPS
#
# Purpose
# -------
# • Visualize relationships between all variables
# • Compare Pearson vs Spearman correlations
# • Automatically identify strongest positive and
#   negative relationships
#
# ===============================================================

# ---------------------------------------------------------------
# PEARSON CORRELATION MATRIX
# ---------------------------------------------------------------

pearson_corr = df[numeric_columns].corr(method="pearson")

pearson_corr.to_csv(
    "Output/Bivariate Analysis/pearson_correlation.csv"
)

# ---------------------------------------------------------------
# SPEARMAN CORRELATION MATRIX
#
# Spearman measures monotonic relationships and is
# less affected by outliers than Pearson.
# ---------------------------------------------------------------

spearman_corr = df[numeric_columns].corr(method="spearman")

spearman_corr.to_csv(
    "Output/Bivariate Analysis/spearman_correlation.csv"
)

print("Correlation matrices exported.")

# ===============================================================
# PEARSON HEATMAP
# ===============================================================

plt.figure(figsize=(18,15))

plt.imshow(
    pearson_corr,
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(pearson_corr.columns)),
    pearson_corr.columns,
    rotation=90,
    fontsize=8
)

plt.yticks(
    range(len(pearson_corr.columns)),
    pearson_corr.columns,
    fontsize=8
)

plt.title("Pearson Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "Graphs/Bivariate Analysis/Correlation/pearson_heatmap.png",
    dpi=300
)

plt.close()

# ===============================================================
# SPEARMAN HEATMAP
# ===============================================================

plt.figure(figsize=(18,15))

plt.imshow(
    spearman_corr,
    aspect="auto"
)

# Reset matplotlib configuration to default to avoid carryover styles
plt.rcParams.update(plt.rcParamsDefault)

plt.colorbar()

plt.xticks(
    range(len(spearman_corr.columns)),
    spearman_corr.columns,
    rotation=90,
    fontsize=8
)

plt.yticks(
    range(len(spearman_corr.columns)),
    spearman_corr.columns,
    fontsize=8
)

plt.title("Spearman Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "Graphs/Bivariate Analysis/Correlation/spearman_heatmap.png",
    dpi=300
)

plt.close()

print("Heatmaps created.")

# ===============================================================
# FIND STRONGEST CORRELATIONS
#
# Ignore self correlations (=1)
# ===============================================================

pairs = []

for i in range(len(pearson_corr.columns)):
    for j in range(i + 1, len(pearson_corr.columns)):
        pairs.append({
            "Variable 1": pearson_corr.columns[i],
            "Variable 2": pearson_corr.columns[j],
            "Pearson":
            pearson_corr.iloc[i, j],
            "Spearman":
            spearman_corr.iloc[i, j]
        })

pairs = pd.DataFrame(pairs)

# ===============================================================
# STRONGEST POSITIVE
# ===============================================================

positive = (
    pairs
    .sort_values(
        by="Pearson",
        ascending=False
    )
)

positive.to_csv(
    "Output/Bivariate Analysis/strongest_positive_correlations.csv",
    index=False
)

print("\nTop Positive Correlations\n")
print(
    positive.head(20)
)

# ===============================================================
# STRONGEST NEGATIVE
# ===============================================================

negative = (
    pairs
    .sort_values(
        by="Pearson"
    )
)

negative.to_csv(
    "Output/Bivariate Analysis/strongest_negative_correlations.csv",
    index=False
)

print("\nTop Negative Correlations\n")
print(
    negative.head(20)
)

# ===============================================================
# AUTOMATIC INTERPRETATION
#
# Generates a simple interpretation of each pair.
# ===============================================================

interpretations = []

for _, row in pairs.iterrows():
    r = row["Pearson"]
    if abs(r) >= 0.90:
        strength = "Very Strong"
    elif abs(r) >= 0.70:
        strength = "Strong"
    elif abs(r) >= 0.50:
        strength = "Moderate"
    elif abs(r) >= 0.30:
        strength = "Weak"
    else:
        strength = "Very Weak"

    if r > 0:
        direction = "Positive"
    elif r < 0:
        direction = "Negative"
    else:
        direction = "No"

    interpretations.append({
        "Variable 1": row["Variable 1"],
        "Variable 2": row["Variable 2"],
        "Pearson": round(r,3),
        "Spearman": round(row["Spearman"],3),
        "Relationship": f"{strength} {direction}"
    })

interpretations = pd.DataFrame(interpretations)

interpretations.to_csv(
    "Output/Bivariate Analysis/correlation_interpretation.csv",
    index=False
)

print("\nInterpretation table exported.")

# ===============================================================
# TOP 15 POSITIVE CORRELATIONS BAR CHART
# ===============================================================

top15 = positive.head(15)

labels = [
    a + "\n&\n" + b
    for a, b in zip(
        top15["Variable 1"],
        top15["Variable 2"]
    )
]

plt.figure(figsize=(14,7))

plt.bar(
    labels,
    top15["Pearson"]
)

plt.xticks(rotation=90)
plt.ylabel("Pearson Correlation")
    
plt.title("Top 15 Strongest Positive Relationships")
plt.tight_layout()
plt.savefig(
    "Graphs/Bivariate Analysis/Correlation/top15_positive.png",
    dpi=300
)
plt.close()

# ===============================================================
# TOP 15 NEGATIVE CORRELATIONS BAR CHART
# ===============================================================

bottom15 = negative.head(15)

labels = [
    a + "\n&\n" + b
    for a, b in zip(
        bottom15["Variable 1"],
        bottom15["Variable 2"]
    )
]

plt.figure(figsize=(14,7))

plt.bar(
    labels,
    bottom15["Pearson"]
)

plt.xticks(rotation=90)
plt.ylabel("Pearson Correlation")
plt.title("Top 15 Strongest Negative Relationships")
plt.tight_layout()
plt.savefig(
    "Graphs/Bivariate Analysis/Correlation/top15_negative.png",
    dpi=300
)
plt.close()

print("\nCorrelation analysis complete.")