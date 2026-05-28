import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load Dataset
diabetes = pd.read_csv("diabetes_dataset.csv")

# Remove missing values
diabetes = diabetes.dropna()

# Feature Selection
X = diabetes.drop(columns=["Outcome"])

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

diabetes["Cluster"] = kmeans.fit_predict(X_scaled)

print(diabetes[["Outcome", "Cluster"]].head(10))

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotting
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=diabetes["Cluster"],
    cmap="viridis"
)

# Centroids
centroids = pca.transform(kmeans.cluster_centers_)

plt.scatter(
    centroids[:,0],
    centroids[:,1],
    c="red",
    marker="X",
    s=200,
    label="Centroids"
)

plt.title("K-Means Clustering on Diabetes Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)

plt.show()