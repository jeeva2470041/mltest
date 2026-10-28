import json

code = """# ============ CLUSTERING: K-MEANS (PCA vs WITHOUT PCA) ============

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_rel

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

# ================= 1. LOAD DATASET =================
df = pd.read_csv("dataset.csv")            # Change to your dataset file
df = df.drop_duplicates()
df = df.drop(columns=["id", "Record_ID"], errors="ignore")

# ================= 2. EDA =================
print(df.info())
print(df.describe(include="all"))
print("\\nMissing values:\\n", df.isnull().sum())

df.select_dtypes(include=np.number).hist(figsize=(10, 6))
plt.tight_layout()
plt.show()

sns.heatmap(df.select_dtypes(include=np.number).corr(),
            cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# ================= 3. FEATURES SELECTION =================
# If dataset contains a ground-truth label/target, drop it for unsupervised clustering
target_col = "target"                      # Set or keep None if no target
if target_col in df.columns:
    X = df.drop(columns=[target_col])
else:
    X = df.copy()

num = X.select_dtypes(include=np.number).columns.tolist()
cat = X.select_dtypes(exclude=np.number).columns.tolist()

# ================= 4. PREPROCESSING =================
prep = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False))
    ]), cat)
])

# Fit and transform features for cluster optimization
X_prep = prep.fit_transform(X)

# ================= 5. ELBOW METHOD & SILHOUETTE ANALYSIS =================
k_range = range(2, 11)
inertias = []
silhouette_scores = []

# Inertia for k=1
inertia_k1 = KMeans(n_clusters=1, random_state=42, n_init=10).fit(X_prep).inertia_
all_k = [1] + list(k_range)
all_inertias = [inertia_k1]

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_prep)
    inertias.append(km.inertia_)
    all_inertias.append(km.inertia_)
    score = silhouette_score(X_prep, labels)
    silhouette_scores.append(score)

# Plot 1: Elbow Curve (Inertia vs k)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(all_k, all_inertias, marker="o", color="b")
plt.title("Elbow Method (Inertia vs k)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.grid(True)

# Plot 2: Silhouette Score vs k
plt.subplot(1, 2, 2)
plt.plot(list(k_range), silhouette_scores, marker="s", color="r")
plt.title("Silhouette Analysis (Score vs k)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)

plt.tight_layout()
plt.show()

# Select best k automatically based on highest silhouette score
best_k = list(k_range)[np.argmax(silhouette_scores)]
print(f"\\nOptimal k selected (by Silhouette Score): {best_k}")
print(f"Best Silhouette Score: {max(silhouette_scores):.4f}")

# ================= 6. MODELS =================
# Model 1: K-Means without PCA
pipe1 = Pipeline([
    ("prep", prep),
    ("kmeans", KMeans(n_clusters=best_k, random_state=42, n_init=10))
])

# Model 2: K-Means with PCA (retaining 95% variance)
pipe2 = Pipeline([
    ("prep", prep),
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("kmeans", KMeans(n_clusters=best_k, random_state=42, n_init=10))
])

# Fit both models
pipe1.fit(X)
pipe2.fit(X)

# ================= 7. PERFORMANCE METRICS =================
# Extract cluster labels & representations
labels1 = pipe1.named_steps["kmeans"].labels_
labels2 = pipe2.named_steps["kmeans"].labels_

X_no_pca = pipe1.named_steps["prep"].transform(X)
X_pca = pipe2.named_steps["pca"].transform(X_no_pca)

results = [
    ("Without PCA", pipe1, X_no_pca, labels1),
    ("With PCA", pipe2, X_pca, labels2)
]

for name, pipe, X_transformed, labels in results:
    km = pipe.named_steps["kmeans"]
    sil = silhouette_score(X_transformed, labels)
    db = davies_bouldin_score(X_transformed, labels)
    ch = calinski_harabasz_score(X_transformed, labels)

    print(f"\\n================ {name} ================")
    print(f"Number of Clusters (k): {best_k}")
    print(f"Silhouette Score (higher is better): {sil:.4f}")
    print(f"Davies-Bouldin Score (lower is better): {db:.4f}")
    print(f"Calinski-Harabasz Score (higher is better): {ch:.4f}")
    print(f"Inertia: {km.inertia_:.2f}")

    if "pca" in pipe.named_steps:
        pca = pipe.named_steps["pca"]
        print(f"Original Dimensions: {X_no_pca.shape[1]}")
        print(f"PCA Reduced Dimensions: {pca.n_components_}")
        print(f"Total Explained Variance: {pca.explained_variance_ratio_.sum():.4f}")

# ================= 8. CLUSTER VISUALIZATION & PROFILING =================
# 2D PCA projection for visualizing clusters
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_no_pca)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels1, palette="tab10", s=50)
plt.title(f"K-Means Clusters Without PCA (k={best_k})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels2, palette="tab10", s=50)
plt.title(f"K-Means Clusters With PCA (k={best_k})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")

plt.tight_layout()
plt.show()

# Cluster Distribution
df_clustered = X.copy()
df_clustered["Cluster_Without_PCA"] = labels1
df_clustered["Cluster_With_PCA"] = labels2

print("\\nCluster Distribution (Without PCA):\\n", df_clustered["Cluster_Without_PCA"].value_counts().sort_index())
print("\\nCluster Distribution (With PCA):\\n", df_clustered["Cluster_With_PCA"].value_counts().sort_index())

# ================= 9. 5-FOLD CV & PAIRED T-TEST =================
# Evaluate statistical significance across 5 folds using Silhouette Score
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores_no_pca = []
scores_pca = []

for train_idx, val_idx in cv.split(X):
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]

    # Model 1: Without PCA
    p1 = Pipeline([
        ("prep", prep),
        ("kmeans", KMeans(n_clusters=best_k, random_state=42, n_init=10))
    ])
    p1.fit(X_train_cv)
    val_prep1 = p1.named_steps["prep"].transform(X_val_cv)
    val_labels1 = p1.named_steps["kmeans"].predict(val_prep1)
    scores_no_pca.append(silhouette_score(val_prep1, val_labels1))

    # Model 2: With PCA
    p2 = Pipeline([
        ("prep", prep),
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("kmeans", KMeans(n_clusters=best_k, random_state=42, n_init=10))
    ])
    p2.fit(X_train_cv)
    val_prep2 = p2.named_steps["prep"].transform(X_val_cv)
    val_pca2 = p2.named_steps["pca"].transform(val_prep2)
    val_labels2 = p2.named_steps["kmeans"].predict(val_pca2)
    scores_pca.append(silhouette_score(val_pca2, val_labels2))

scores_no_pca = np.array(scores_no_pca)
scores_pca = np.array(scores_pca)

print("\\nFold-wise Validation Silhouette Scores:")
fold_df = pd.DataFrame({
    "Fold": range(1, 6),
    "Without PCA": scores_no_pca,
    "With PCA": scores_pca,
    "Difference": scores_no_pca - scores_pca
})
print(fold_df.to_string(index=False))

print(f"\\nMean Silhouette (Without PCA): {scores_no_pca.mean():.4f} +/- {scores_no_pca.std():.4f}")
print(f"Mean Silhouette (With PCA):    {scores_pca.mean():.4f} +/- {scores_pca.std():.4f}")

# Paired t-test
t_stat, p_value = ttest_rel(scores_no_pca, scores_pca)

print("\\nSilhouette Score Paired t-test")
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Significant difference in clustering quality (p < 0.05)")
else:
    print("No significant difference in clustering quality (p >= 0.05)")
"""

lines = [l + "\n" for l in code.split("\n")]
# Remove last trailing newline
if lines and lines[-1] == "\n":
    lines[-1] = ""

nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("c:/Users/priya/jeeva_project/mltest/complete-code/kmeans.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Created kmeans.ipynb successfully")

