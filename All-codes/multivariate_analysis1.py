"""
===============================================================
08_multivariate_analysis.py

Purpose
Analyse how multiple macroeconomic variables interact simultaneously.

Topics
1. Multiple Regression (with Feature Contribution plots)
2. Principal Component Analysis (with Biplots)
3. K-Means Clustering & Structural Country Profiling
===============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set style for high-quality publication graphics
sns.set_theme(style="whitegrid")

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================
folders = [
    "Output/Multivariate Analysis",
    "Graphs/Multivariate Analysis",
    "Graphs/Multivariate Analysis/Regression",
    "Graphs/Multivariate Analysis/PCA",
    "Graphs/Multivariate Analysis/Clusters"
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
# FIND IMPORTANT VARIABLES
# ============================================================
def find_column(keyword):
    for column in df.columns:
        if keyword.lower() in column.lower():
            return column
    return None

gdp_col = find_column("gross_domestic_product")
population_col = find_column("population")
exports_col = find_column("exports")
imports_col = find_column("imports")
investment_col = find_column("gross_capital_formation")
gni_col = find_column("gross_national_income")

# ============================================================
# PART 1: OLS MULTIPLE LINEAR REGRESSION
# ============================================================
print("\n" + "="*70)
print("PART 1: MULTIPLE LINEAR REGRESSION")
print("="*70)

candidate_features = [
    population_col, exports_col, imports_col, investment_col,
    "trade_openness", "investment_ratio", "manufacturing_share",
    "services_share", "agriculture_share"
]

features = [f for f in candidate_features if f in df.columns]

# Ensure we have clean rows without missing target or features
regression_data = df[[gdp_col] + features].dropna()
X = regression_data[features]
y = regression_data[gdp_col]

print(f"Regression Rows Evaluated: {regression_data.shape[0]}")

# Fit OLS Model
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

r2 = r2_score(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
print(f"Model R² : {round(r2, 4)}")
print(f"Model RMSE: {round(rmse, 2)}")

# Save Coefficients & Features Importance
coefficients = pd.DataFrame({
    "Variable": features,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

coefficients.to_csv("Output/Multivariate Analysis/regression_coefficients.csv", index=False)

# Save Model Output Predictions
results = regression_data.copy()
results["Predicted_GDP"] = predictions
results["Residual"] = y - predictions
results.to_csv("Output/Multivariate Analysis/regression_predictions.csv", index=False)

# Plot 1: Actual vs Predicted Scatter
plt.figure(figsize=(8, 6))
plt.scatter(y, predictions, alpha=0.6, color=sns.color_palette("viridis")[2])
ideal_line = [min(y.min(), predictions.min()), max(y.max(), predictions.max())]
plt.plot(ideal_line, ideal_line, color="red", linestyle="--", linewidth=2, label="Perfect Model Line")
plt.xlabel("Actual GDP")
plt.ylabel("Predicted GDP")
plt.title(f"Multiple Regression Performance (R² = {round(r2, 3)})")
plt.legend()
plt.tight_layout()
plt.savefig("Graphs/Multivariate Analysis/Regression/actual_vs_predicted.png", dpi=300)
plt.close()

# Plot 2: Regression Coefficients Bar Chart
plt.figure(figsize=(10, 5))
sns.barplot(x="Coefficient", y="Variable", data=coefficients, palette="vlag")
plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
plt.title("Macroeconomic Feature Contribution on Target GDP")
plt.tight_layout()
plt.savefig("Graphs/Multivariate Analysis/Regression/regression_feature_importance.png", dpi=300)
plt.close()

# ============================================================
# PART 2: PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================
print("\n" + "="*70)
print("PART 2: PRINCIPAL COMPONENT ANALYSIS")
print("="*70)

candidate_columns = [
    gdp_col, gni_col, population_col, exports_col, imports_col, investment_col,
    "trade_openness", "investment_ratio", "manufacturing_share",
    "services_share", "agriculture_share", "government_share",
    "household_share", "gdp_per_person"
]
pca_columns = [col for col in candidate_columns if col in df.columns]

pca_data = df[pca_columns].dropna()

# Standardize variables safely
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pca_data)

# Fit PCA
pca = PCA()
principal_components = pca.fit_transform(scaled_data)

# Process Variance Dataframe
variance = pd.DataFrame({
    "Principal Component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "Explained Variance": pca.explained_variance_ratio_,
    "Cumulative Variance": np.cumsum(pca.explained_variance_ratio_)
})
variance.to_csv("Output/Multivariate Analysis/pca_variance.csv", index=False)

# Scree Plot with Cumulative Line
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.bar(range(1, len(variance)+1), variance["Explained Variance"], alpha=0.6, color="g", label="Individual Variance")
ax1.set_xlabel("Principal Components")
ax1.set_ylabel("Individual Explained Variance Ratio")
ax1.set_xticks(range(1, len(variance)+1))

ax2 = ax1.twinx()
ax2.plot(range(1, len(variance)+1), variance["Cumulative Variance"], marker="o", color="b", label="Cumulative Variance")
ax2.set_ylabel("Cumulative Variance Ratio")
plt.title("PCA Scree & Variance Accrual Plot")
fig.tight_layout()
plt.savefig("Graphs/Multivariate Analysis/PCA/scree_plot.png", dpi=300)
plt.close()

# Save PCA Structural Metadata
loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(len(pca_columns))], index=pca_columns)
loadings.to_csv("Output/Multivariate Analysis/pca_loadings.csv")

scores = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(len(pca_columns))])
scores["country"] = df.loc[pca_data.index, "country"].values
scores["year"] = df.loc[pca_data.index, "year"].values
scores.to_csv("Output/Multivariate Analysis/pca_scores.csv", index=False)

# Enhanced PCA Space Scatter (Biplot-inspired)
plt.figure(figsize=(10, 7))
sns.scatterplot(x="PC1", y="PC2", data=scores, alpha=0.7, color="indigo")
# Overlay vector axes representing original variables
for i, feature_name in enumerate(pca_columns):
    plt.arrow(0, 0, loadings.iloc[i, 0]*5, loadings.iloc[i, 1]*5, color='red', alpha=0.6, head_width=0.15)
    plt.text(loadings.iloc[i, 0]*6, loadings.iloc[i, 1]*6, feature_name, color='brown', ha='center', va='center', fontsize=8)
plt.xlabel(f"PC1 ({round(variance.iloc[0,1]*100, 1)}% Variance)")
plt.ylabel(f"PC2 ({round(variance.iloc[1,1]*100, 1)}% Variance)")
plt.title("Global Macroeconomic Structural Space (PCA Biplot)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Graphs/Multivariate Analysis/PCA/pca_scatter.png", dpi=300)
plt.close()

print("PCA Matrix Completed.")

# ============================================================
# PART 3: K-MEANS CLUSTERING & PROFILING
# ============================================================
print("\n" + "="*70)
print("PART 3: K-MEANS CLUSTERING")
print("="*70)

cluster_data = scaled_data
wcss = []
for k_val in range(1, 11):
    km_model = KMeans(n_clusters=k_val, random_state=42, n_init=20)
    km_model.fit(cluster_data)
    wcss.append(km_model.inertia_)

# Elbow Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o", color="darkorange", linewidth=2)
plt.xlabel("Number of Structural Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.title("Optimal Structural Segregation (Elbow Method)")
plt.tight_layout()
plt.savefig("Graphs/Multivariate Analysis/Clusters/elbow_method.png", dpi=300)
plt.close()

# Fit Final Cluster Assignments
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
clusters = kmeans.fit_predict(cluster_data)

scores["Cluster"] = clusters
scores.to_csv("Output/Multivariate Analysis/pca_clusters.csv", index=False)

# Plot Clusters in Reduced Variance Coordinates
plt.figure(figsize=(11, 8))
sns.scatterplot(x="PC1", y="PC2", hue="Cluster", palette="Set1", data=scores, alpha=0.8, s=60)
plt.title("Global Structural Country Groups inside Component Domain")
plt.tight_layout()
plt.savefig("Graphs/Multivariate Analysis/Clusters/pca_clusters.png", dpi=300)
plt.close()

# Save Country Profiles Mapping Tables
clustered_df = df.loc[pca_data.index].copy()
clustered_df["Cluster"] = clusters
clustered_df.to_csv("Output/Multivariate Analysis/clustered_dataset.csv", index=False)

# Profiling Structural Metrics Matrix
cluster_summary = clustered_df.groupby("Cluster")[pca_columns].mean()
cluster_summary.to_csv("Output/Multivariate Analysis/cluster_summary.csv")

cluster_counts = clustered_df.groupby("Cluster")["country"].nunique().reset_index(name="Unique Countries")
cluster_counts.to_csv("Output/Multivariate Analysis/cluster_country_counts.csv", index=False)

print("\n--- Structural Macro Profiles Matrix (Cluster Averages) ---")
print(cluster_summary.T)

print("\n--- Country Allocations Per Cluster Portfolio ---")
print(cluster_counts)

# Cross-tabulations with Economic Classification Categories
if "income_group" in clustered_df.columns:
    income_dist = pd.crosstab(clustered_df["Cluster"], clustered_df["income_group"])
    income_dist.to_csv("Output/Multivariate Analysis/income_cluster_distribution.csv")

print("\nMultivariate Script Pipeline Executed Successfully.")