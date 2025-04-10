import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json

# Load data
df = pd.read_csv("datasets/squid_orders_export.csv")
df.fillna(0, inplace=True)

# Add derived features
df["spread"] = df["ask_price_1"] - df["bid_price_1"]
df["volume_imbalance"] = (
    df["bid_volume_1"] + df["bid_volume_2"] + df["bid_volume_3"]
    - df["ask_volume_1"] - df["ask_volume_2"] - df["ask_volume_3"]
)
df["total_volume"] = (
    df["bid_volume_1"] + df["bid_volume_2"] + df["bid_volume_3"]
    + df["ask_volume_1"] + df["ask_volume_2"] + df["ask_volume_3"]
)
df["volume_ratio"] = df["volume_imbalance"] / df["total_volume"]
df["price_to_mid_bid"] = df["mid_price"] - df["bid_price_1"]
df["price_to_mid_ask"] = df["ask_price_1"] - df["mid_price"]
df["spread_ratio"] = df["spread"] / df["mid_price"]

# df["price_order_ratio"] = df["bid_ask"]

# Features for clustering
features = [
    "bid_volume_1", "bid_volume_2", "bid_volume_3", 
    "ask_volume_1","ask_volume_2", "ask_volume_3",
    "volume_ratio", "price_to_mid_bid", "price_to_mid_ask"
]
X = df[features]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for 2D visualization
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)


# Plot clusters in 2D
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["cluster"], cmap="tab10", alpha=0.6)
plt.title("K-Means Clustering of Orders with Enhanced Features")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

# Lookahead analysis: how does mid_price change 50 steps later?
lookahead_steps = 50
df.sort_values(by="timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)
df["mid_price_future"] = df["mid_price"].shift(-lookahead_steps)
df["mid_price_change"] = df["mid_price_future"] - df["mid_price"]

# Group by cluster and calculate mean price impact
impact_by_cluster = df.groupby("cluster")["mid_price_change"].mean()

# Plot average price impact per cluster
plt.figure(figsize=(8, 5))
impact_by_cluster.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title(f"Average Mid Price Change {lookahead_steps} Steps Ahead by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Mid Price Change")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

params = {
    "feature_names": features,
    "means": scaler.mean_.tolist(),
    "stds": scaler.scale_.tolist(),
    "cluster_centers": kmeans.cluster_centers_.tolist()
}

# Save to file or print
with open("cluster_model_params.json", "w") as f:
    json.dump(params, f, indent=2)

print("Saved means, stds, and cluster centers to cluster_model_params.json")