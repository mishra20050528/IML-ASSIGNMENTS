import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load Dataset
social = pd.read_csv("Social_Network_Ads.csv")

# Drop unnecessary column
social = social.drop(columns=["User ID"])

# Convert categorical data
social["Gender"] = social["Gender"].map({
    "Male": 0,
    "Female": 1
})

# Remove missing values
social = social.dropna()

# Feature Selection
X = social.drop(columns=["Purchased"])

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

social["Cluster"] = kmeans.fit_predict(X_scaled)

print(social[["Purchased", "Cluster"]].head(10))

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotting
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=social["Cluster"],
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

plt.title("K-Means Clustering on Social Network Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)

plt.show()