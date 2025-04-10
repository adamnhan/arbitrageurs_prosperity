import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the flattened CSV
df = pd.read_csv('flattened_high_volume_orders.csv')

# One-hot encode the 'side' column
df_encoded = pd.get_dummies(df, columns=['side'])

# Select features for clustering
features = ['price', 'volume', 'mid_price'] + [col for col in df_encoded.columns if col.startswith('side_')]
X = df_encoded[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Optional: visualize with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='tab10', s=10)
plt.title('K-Means Clustering of Orders (PCA Reduced)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()

# Save clustered data
df.to_csv('clustered_high_volume_orders.csv', index=False)

print("Clustered data saved to 'clustered_high_volume_orders.csv'")
