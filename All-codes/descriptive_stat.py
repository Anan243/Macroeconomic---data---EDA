"""
===========================================================

04_descriptive_statistics.py

Global Macroeconomic Structural Analysis

Purpose

Perform descriptive statistical analysis before
moving to visualization.

Economists always inspect

• distribution
• centre
• spread
• skewness
• variability
• missing values

before interpreting data.

===========================================================
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

folders = [

    "Output/Descriptive Statistics",
    "Graphs/Descriptive Statistics"

]

for folder in folders:

    os.makedirs(folder, exist_ok=True)

# ============================================================
# LOAD FEATURE ENGINEERED DATASET
# ============================================================

df = pd.read_csv(

    "Output/Feature Engineering/feature_engineered_dataset.csv"

)

print("\nDataset Loaded")

print(df.shape)

# ============================================================
# IDENTIFY NUMERIC COLUMNS
# ============================================================

numeric_columns = df.select_dtypes(

    include=np.number

).columns.tolist()

print("\nNumeric Variables\n")

for column in numeric_columns:

    print(column)

print(

    "\nTotal Numeric Variables :",

    len(numeric_columns)

)

# ============================================================
# BASIC DESCRIPTIVE STATISTICS
# ============================================================

summary = df[numeric_columns].describe().T

summary.rename(

    columns={

        "count":"Count",
        "mean":"Mean",
        "std":"Standard Deviation",
        "min":"Minimum",
        "25%":"Q1",
        "50%":"Median",
        "75%":"Q3",
        "max":"Maximum"

    },

    inplace=True

)

# ============================================================
# RANGE
# ============================================================

summary["Range"] = (

    summary["Maximum"]

    -

    summary["Minimum"]

)

# ============================================================
# INTERQUARTILE RANGE
# ============================================================

summary["IQR"] = (

    summary["Q3"]

    -

    summary["Q1"]

)

# ============================================================
# COEFFICIENT OF VARIATION
#
# Standard Deviation / Mean
#
# Measures relative variability.
#
# Higher values indicate greater dispersion
# relative to the average.
# ============================================================

summary["Coefficient_of_Variation"] = (

    summary["Standard Deviation"]

    /

    summary["Mean"]

)

# ============================================================
# SKEWNESS
#
# Positive
#
# Long right tail
#
# Negative
#
# Long left tail
#
# Zero
#
# Approximately symmetric
# ============================================================

summary["Skewness"] = (

    df[numeric_columns]

    .skew()

)

# ============================================================
# KURTOSIS
#
# Indicates heaviness of tails.
# ============================================================

summary["Kurtosis"] = (

    df[numeric_columns]

    .kurtosis()

)

# ============================================================
# MISSING VALUES
# ============================================================

summary["Missing Values"] = (

    df[numeric_columns]

    .isna()

    .sum()

)

summary["Missing Percentage"] = (

    (

        df[numeric_columns]

        .isna()

        .mean()

    )

    *100

).round(2)

# ============================================================
# SAVE SUMMARY TABLE
# ============================================================

summary.to_csv(

    "Output/Descriptive Statistics/descriptive_statistics.csv"

)

print("\nDescriptive Statistics Saved")

print(summary.head())

# ============================================================
# PRINT IMPORTANT INFORMATION
# ============================================================

print("\n")

print("="*60)

print("VARIABLE SUMMARY")

print("="*60)

for column in numeric_columns:

    print("\n")

    print(column)

    print("-------------------------")

    print(

        "Mean :", round(df[column].mean(),3)

    )

    print(

        "Median :", round(df[column].median(),3)

    )

    print(

        "Std Dev :", round(df[column].std(),3)

    )

    print(

        "Minimum :", round(df[column].min(),3)

    )

    print(

        "Maximum :", round(df[column].max(),3)

    )

    print(

        "Skewness :", round(df[column].skew(),3)

    )

    print(

        "Missing :", df[column].isna().sum()

    )

print("\nBasic Descriptive Analysis Complete.")


# ============================================================
# PART 2
#
# VISUAL DISTRIBUTION ANALYSIS
#
# Economists never rely only on summary statistics.
#
# We visualize distributions because two variables can
# have the same mean but very different distributions.
#
# ============================================================

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Create graph folders
# ------------------------------------------------------------

graph_folders = [

    "Graphs/Descriptive Statistics/Histograms",
    "Graphs/Descriptive Statistics/Boxplots",
    "Graphs/Descriptive Statistics/DensityPlots"

]

for folder in graph_folders:

    os.makedirs(folder, exist_ok=True)

# ============================================================
# HISTOGRAMS
#
# Economic Question:
#
# How are countries distributed?
#
# Example:
#
# GDP is usually right-skewed because
# only a few countries produce extremely
# large output.
#
# ============================================================

print("\nGenerating Histograms...")

for column in numeric_columns:

    plt.figure(figsize=(8,5))

    plt.hist(

        df[column].dropna(),

        bins=30,

        edgecolor="black"

    )

    plt.title(f"Histogram of {column}")

    plt.xlabel(column)

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(

        f"Graphs/Descriptive Statistics/Histograms/{column}.png",

        dpi=300

    )

    plt.close()

print("Histograms Created.")

# ============================================================
# BOXPLOTS
#
# Economic Question
#
# Which countries are outliers?
#
# GDP:
#
# USA
#
# China
#
# are expected to appear as outliers.
#
# ============================================================

print("\nGenerating Boxplots...")

for column in numeric_columns:

    plt.figure(figsize=(8,4))

    plt.boxplot(

        df[column].dropna(),

        vert=False

    )

    plt.title(f"Boxplot of {column}")

    plt.xlabel(column)

    plt.tight_layout()

    plt.savefig(

        f"Graphs/Descriptive Statistics/Boxplots/{column}.png",

        dpi=300

    )

    plt.close()

print("Boxplots Created.")

# ============================================================
# DENSITY PLOTS
#
# Gives smoother distribution than histogram.
#
# Useful for detecting
#
# Multi-modal distributions.
#
# ============================================================

print("\nGenerating Density Plots...")

for column in numeric_columns:

    plt.figure(figsize=(8,5))

    df[column].dropna().plot(

        kind="density"

    )

    plt.title(f"Density Plot of {column}")

    plt.xlabel(column)

    plt.tight_layout()

    plt.savefig(

        f"Graphs/Descriptive Statistics/DensityPlots/{column}.png",

        dpi=300

    )

    plt.close()

print("Density Plots Created.")

# ============================================================
# PART 3
#
# OUTLIER ANALYSIS
# NORMALITY ANALYSIS
# VARIABILITY ANALYSIS
#
# Purpose
# -------
# This section identifies:
#
# • Outliers
# • Skewness
# • Distribution Shape
# • Relative Variability
#
# ============================================================

print("\n")
print("="*70)
print("OUTLIER ANALYSIS")
print("="*70)

outlier_summary = []

for column in numeric_columns:

    data = df[column].dropna()

    if len(data) == 0:
        continue

    # ----------------------------
    # Quartiles
    # ----------------------------

    Q1 = data.quantile(0.25)

    Q3 = data.quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - (1.5 * IQR)

    upper = Q3 + (1.5 * IQR)

    outliers = data[
        (data < lower) |
        (data > upper)
    ]

    outlier_summary.append({

        "Variable": column,

        "Lower Bound": lower,

        "Upper Bound": upper,

        "Number of Outliers": len(outliers),

        "Percentage":

        round(

            len(outliers)

            /

            len(data)

            *

            100,

            2

        )

    })

outlier_summary = pd.DataFrame(outlier_summary)

outlier_summary.to_csv(

    "Output/Descriptive Statistics/outlier_summary.csv",

    index=False

)

print(outlier_summary)

# ============================================================
# SKEWNESS INTERPRETATION
#
# Interpretation
#
# -0.5 to 0.5
#
# Approximately Symmetric
#
# 0.5 to 1
#
# Moderately Right Skewed
#
# >1
#
# Highly Right Skewed
#
# Similar for negative values.
# ============================================================

print("\n")
print("="*70)
print("SKEWNESS INTERPRETATION")
print("="*70)

skewness_results = []

for column in numeric_columns:

    skew = df[column].skew()

    if skew > 1:

        interpretation = "Highly Right Skewed"

    elif skew > 0.5:

        interpretation = "Moderately Right Skewed"

    elif skew >= -0.5:

        interpretation = "Approximately Symmetric"

    elif skew >= -1:

        interpretation = "Moderately Left Skewed"

    else:

        interpretation = "Highly Left Skewed"

    skewness_results.append({

        "Variable": column,

        "Skewness": round(skew,3),

        "Interpretation": interpretation

    })

skewness_results = pd.DataFrame(

    skewness_results

)

print(skewness_results)

skewness_results.to_csv(

    "Output/Descriptive Statistics/skewness_summary.csv",

    index=False

)

# ============================================================
# COEFFICIENT OF VARIATION
#
# Measures variability relative to the mean.
#
# Economists use this when variables
# have different scales.
# ============================================================

print("\n")
print("="*70)
print("COEFFICIENT OF VARIATION")
print("="*70)

cv_summary = []

for column in numeric_columns:

    mean = df[column].mean()

    std = df[column].std()

    if mean == 0:

        continue

    cv = std / mean

    cv_summary.append({

        "Variable": column,

        "Coefficient of Variation":

        round(cv,3)

    })

cv_summary = (

    pd.DataFrame(cv_summary)

    .sort_values(

        by="Coefficient of Variation",

        ascending=False

    )

)

print(cv_summary)

cv_summary.to_csv(

    "Output/Descriptive Statistics/coefficient_of_variation.csv",

    index=False

)

# ============================================================
# AUTOMATIC INTERPRETATION
#
# Generates simple observations that can later
# be included in the final report.
# ============================================================

print("\n")
print("="*70)
print("AUTOMATIC INTERPRETATION")
print("="*70)

interpretation = []

for column in numeric_columns:

    mean = df[column].mean()

    median = df[column].median()

    skew = df[column].skew()

    if skew > 1:

        text = (

            f"{column} has a highly right-skewed "

            "distribution. "

            "A small number of countries have "

            "exceptionally high values."

        )

    elif skew < -1:

        text = (

            f"{column} has a highly left-skewed "

            "distribution."

        )

    else:

        text = (

            f"{column} is relatively balanced "

            "across countries."

        )

    interpretation.append({

        "Variable": column,

        "Mean": round(mean,2),

        "Median": round(median,2),

        "Interpretation": text

    })

interpretation = pd.DataFrame(

    interpretation

)

interpretation.to_csv(

    "Output/Descriptive Statistics/automatic_interpretation.csv",

    index=False

)

print(interpretation)

print("\n")
print("="*70)
print("DESCRIPTIVE STATISTICS COMPLETE")
print("="*70)
