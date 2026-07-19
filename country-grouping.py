"""
===============================================================

group_comparison.py

Purpose

Compare macroeconomic indicators across


1. Income Groups
2. Development Status
3. Population Groups
4. GDP Groups

===============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================================================
# CREATE OUTPUT FOLDERS
# ===============================================================

folders = [

    "Output/Group Comparison",
    "Graphs/Group Comparison"

]

for folder in folders:

    os.makedirs(folder, exist_ok=True)


# ===============================================================
# LOAD DATA
# ===============================================================

df = pd.read_csv(
    "Output/Feature Engineering/feature_engineered_dataset.csv"
)

print("="*60)
print("Original Dataset Shape")
print("="*60)
print(df.shape)

# ===============================================================
# KEEP ONLY YEAR 2021
#
# Purpose:
# This analysis compares only those countries using their latest available
# macroeconomic indicators (2021). All earlier years data are removed.
# ===============================================================

df = df[df["year"] == 2021].copy()

print("\n" + "="*60)
print("Dataset Shape After Keeping Only Year 2021")
print("="*60)
print(df.shape)

print("\nYear Distribution After Filtering")
print(df["year"].value_counts().sort_index())

print("\nNumber of Unique Countries")
print(df["country"].nunique())

print("\nFirst Five Rows")
print(df.head())





# ===============================================================
# VARIABLES TO COMPARE
#
# These variables capture different dimensions of
# macroeconomic structure.
# ===============================================================

variables = [

    "trade_openness",

    "investment_ratio",

    "government_share",

    "household_share",

    "manufacturing_share",

    "agriculture_share",

    "services_share",

    "export_dependency",

    "import_dependency"

]

# Keep only variables that actually exist

variables = [

    v

    for v in variables

    if v in df.columns

]

print("\nVariables Used\n")

for v in variables:

    print(v)

# ===============================================================
# HELPER FUNCTION
#
# Calculates summary statistics for every group.
# ===============================================================

def compare_groups(group_column):

    if group_column not in df.columns:

        print(group_column,"not found")

        return

    summary = (

        df

        .groupby(group_column)[variables]

        .agg(

            [

                "mean",

                "median",

                "std",

                "min",

                "max"

            ]

        )

        .round(3)

    )

    filename = (

        "Output/Group Comparison/"

        + group_column

        + "_summary.csv"

    )

    summary.to_csv(

        filename

    )

    print("\n")

    print("="*70)

    print(group_column)

    print("="*70)

    print(summary)


# ===============================================================
# DEVELOPMENT STATUS
# ===============================================================

compare_groups(

    "development_status"

)

# ===============================================================
# INCOME GROUP
# ===============================================================

compare_groups(

    "income_group"

)

# ===============================================================
# GDP GROUP
# ===============================================================

compare_groups(

    "gdp_group"

)

# ===============================================================
# POPULATION GROUP
# ===============================================================

compare_groups(

    "population_group"

)

print("\nSummary Tables Created.")

# ===============================================================
# PART 2
#
# GROUP COMPARISON VISUALIZATIONS
#
# Purpose
# -------
# Visualize how macroeconomic indicators differ across
# continents, income groups and development status.
#
# These graphs are suitable for reports and Power BI.
# ===============================================================

import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Create graph folders
# ---------------------------------------------------------------

graph_folders = [

    "Graphs/Group Comparison/Income Groups",
    "Graphs/Group Comparison/Development Status"

]

for folder in graph_folders:

    os.makedirs(folder, exist_ok=True)

# ===============================================================
# BAR CHARTS
#
# Why?
#
# Compare average values between groups.
#
# ===============================================================

def create_barplots(group_column, folder_name):

    if group_column not in df.columns:
        return

    grouped = (

        df

        .groupby(group_column)[variables]

        .mean()

        .sort_index()

    )

    for variable in variables:

        plt.figure(figsize=(10,6))

        grouped[variable].plot(

            kind="bar"

        )

        plt.title(

            f"{variable.replace('_',' ').title()} by {group_column.replace('_',' ').title()}"

        )

        plt.ylabel(variable)

        plt.xlabel(group_column)

        plt.xticks(rotation=30)

        plt.tight_layout()

        plt.savefig(

            f"Graphs/Group Comparison/{folder_name}/{variable}_bar.png",

            dpi=300

        )

        plt.close()



create_barplots(

    "income_group",

    "Income Groups"

)

create_barplots(

    "development_status",

    "Development Status"

)

print("Bar charts created.")

# ===============================================================
# BOXPLOTS
#
# Why?
#
# Compare distribution within each group.
#
# Example:
#
# Does Europe have similar GDP values?
#
# Or are there huge differences?
#
# ===============================================================

def create_boxplots(group_column, folder_name):

    if group_column not in df.columns:
        return

    for variable in variables:

        plt.figure(figsize=(10,6))

        df.boxplot(

            column=variable,

            by=group_column,

            rot=35,

            grid=False

        )

        plt.title(

            f"{variable.replace('_',' ').title()}"

        )

        plt.suptitle("")

        plt.xlabel(group_column)

        plt.ylabel(variable)

        plt.tight_layout()

        plt.savefig(

            f"Graphs/Group Comparison/{folder_name}/{variable}_boxplot.png",

            dpi=300

        )

        plt.close()



create_boxplots(

    "income_group",

    "Income Groups"

)

create_boxplots(

    "development_status",

    "Development Status"

)

print("Boxplots created.")

# ===============================================================
# LINE CHARTS
#
# Compare group averages.
#
# Easier to compare many indicators at once.
# ===============================================================

def create_lineplots(group_column, folder_name):

    if group_column not in df.columns:
        return

    grouped = (

        df

        .groupby(group_column)[variables]

        .mean()

    )

    for variable in variables:

        plt.figure(figsize=(9,5))

        plt.plot(

            grouped.index.astype(str),

            grouped[variable],

            marker="o"

        )

        plt.title(

            f"{variable.replace('_',' ').title()}"

        )

        plt.xlabel(group_column)

        plt.ylabel(variable)

        plt.xticks(rotation=30)

        plt.tight_layout()

        plt.savefig(

            f"Graphs/Group Comparison/{folder_name}/{variable}_line.png",

            dpi=300

        )

        plt.close()



create_lineplots(

    "income_group",

    "Income Groups"

)

create_lineplots(

    "development_status",

    "Development Status"

)

print("Line charts created.")

# ===============================================================
# EXPORT GROUP AVERAGES
#
# Useful for Power BI and Excel.
# ===============================================================

for group in [

    "income_group",

    "development_status"

]:

    if group in df.columns:

        averages = (

            df

            .groupby(group)[variables]

            .mean()

            .round(3)

        )

        averages.to_csv(

            f"Output/Group Comparison/{group}_means.csv"

        )

print("Average tables exported.")

# ===============================================================
# AUTOMATIC ECONOMIC OBSERVATIONS
#
# Produce a simple summary of which group has the
# highest and lowest average for each indicator.
# ===============================================================

observations = []

for group in [


    "income_group",

    "development_status"

]:

    if group not in df.columns:
        continue

    grouped = (

        df

        .groupby(group)[variables]

        .mean()

    )

    for variable in variables:

        highest = grouped[variable].idxmax()

        lowest = grouped[variable].idxmin()

        observations.append({

            "Grouping": group,

            "Indicator": variable,

            "Highest Average": highest,

            "Lowest Average": lowest,

            "Highest Value": round(grouped[variable].max(),3),

            "Lowest Value": round(grouped[variable].min(),3)

        })

observations = pd.DataFrame(observations)

observations.to_csv(

    "Output/Group Comparison/group_observations.csv",

    index=False

)

print("\nAutomatic observations exported.")

print("\nGroup comparison visualizations complete.")

# ===============================================================
# COUNTRY COUNT BAR CHARTS
#
# Purpose
# -------
# Show how many countries belong to each category.
#
# Unlike the previous bar charts (which plotted averages),
# these graphs plot frequencies (counts).
#
# These graphs answer questions like:
#
# • How many countries are High Income?
# • How many countries are Developing?
# • Which GDP group has the largest number of countries?
#
# ===============================================================

count_folder = "Graphs/Group Comparison/Country Counts"

os.makedirs(count_folder, exist_ok=True)


def create_country_count_plot(group_column):

    if group_column not in df.columns:
        return

    # ----------------------------------------------------------
    # Count number of UNIQUE countries in each group
    #
    # Using nunique() avoids counting the same country multiple
    # times when the dataset contains many years.
    # ----------------------------------------------------------

    counts = (

        df

        .groupby(group_column)["country"]

        .nunique()

        .sort_values(ascending=False)

    )

    print("\n")
    print("="*60)
    print(f"Country Counts : {group_column}")
    print("="*60)
    print(counts)

    plt.figure(figsize=(9,6))

    counts.plot(kind="bar")

    plt.title(

        f"Number of Countries in each {group_column.replace('_',' ').title()}"

    )

    plt.xlabel(

        group_column.replace("_"," ").title()

    )

    plt.ylabel("Number of Countries")

    plt.xticks(rotation=30)

    # ----------------------------------------------------------
    # Write count on top of each bar
    # ----------------------------------------------------------

    for i, value in enumerate(counts):

        plt.text(

            i,

            value + 0.5,

            str(value),

            ha="center"

        )

    plt.tight_layout()

    plt.savefig(

        f"{count_folder}/{group_column}_country_counts.png",

        dpi=300

    )

    plt.close()

    # Export the counts for Power BI / Excel

    counts.to_csv(

        f"Output/Group Comparison/{group_column}_country_counts.csv",

        header=["Number_of_Countries"]

    )

create_country_count_plot("income_group")

create_country_count_plot("development_status")

create_country_count_plot("gdp_group")

create_country_count_plot("population_group")
