import pandas as pd

# Example: load your CSV file.
df = pd.read_csv("datasets/squid_orders_export.csv")

# Display the original dataframe
print("Original DF:")
print(df.head())

# Define the bid and ask columns
bid_cols = ["bid_price_1", "bid_price_2", "bid_price_3", "bid_volume_1", "bid_volume_2", "bid_volume_3"]
ask_cols = ["ask_price_1", "ask_price_2", "ask_price_3", "ask_volume_1", "ask_volume_2", "ask_volume_3"]

# Create separate DataFrames for bids and asks and reset the index to create a unique identifier for each row.
df_bid = df[['timestamp'] + bid_cols].copy().reset_index().rename(columns={'index': 'unique_id'})
df_ask = df[['timestamp'] + ask_cols].copy().reset_index().rename(columns={'index': 'unique_id'})

# Use pd.wide_to_long to melt bid-related columns into long format using the unique id.
df_bid_long = pd.wide_to_long(
    df_bid,
    stubnames=["bid_price", "bid_volume"],
    i="unique_id",
    j="order_level",
    sep="_",
    suffix=r'\d+'
).reset_index()

# Similarly for ask columns.
df_ask_long = pd.wide_to_long(
    df_ask,
    stubnames=["ask_price", "ask_volume"],
    i="unique_id",
    j="order_level",
    sep="_",
    suffix=r'\d+'
).reset_index()

# Optionally, retain the timestamp from each DataFrame. Since unique_id was just for the reshape, you can drop it.
df_bid_long = df_bid_long.drop(columns=['unique_id'])
df_ask_long = df_ask_long.drop(columns=['unique_id'])

# Add a column to indicate the order side.
df_bid_long["side"] = "bid"
df_ask_long["side"] = "ask"

# Rename columns for consistency.
df_bid_long = df_bid_long.rename(columns={"bid_price": "price", "bid_volume": "volume"})
df_ask_long = df_ask_long.rename(columns={"ask_price": "price", "ask_volume": "volume"})

# Concatenate the bid and ask DataFrames.
df_orders = pd.concat([df_bid_long, df_ask_long], ignore_index=True)

# Optionally, sort the resulting DataFrame.
df_orders.sort_values(by=["timestamp", "side", "order_level"], inplace=True)

# Display the transformed DataFrame.
print("Transformed DF (stacked bids and asks):")
print(df_orders.head(10))

# Save to a new CSV file.
df_orders.to_csv("stacked_orders.csv", index=False)

import pandas as pd
import numpy as np

# Load the stacked orders CSV file
df = pd.read_csv("stacked_orders.csv")

# Convert timestamp column to datetime if not already (adjust format if necessary)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ------------------------------------------------------------------------
# 1. Basic Aggregation per Timestamp and Side
# ------------------------------------------------------------------------
# For each timestamp and side (bid or ask), compute:
# - Order count
# - Total volume
# - Mean and standard deviation of prices
# - Minimum and maximum price (these will be useful as best prices)
agg = df.groupby(['timestamp', 'side']).agg({
    'price': ['count', 'mean', 'std', 'min', 'max'],
    'volume': ['sum', 'mean', 'std']
}).reset_index()

# Flatten the column names
agg.columns = ['timestamp', 'side', 'order_count', 'price_mean', 'price_std', 'price_min', 'price_max', 
               'volume_sum', 'volume_mean', 'volume_std']

# ------------------------------------------------------------------------
# 2. Extract Best Bid, Best Ask and Compute Spread Dynamics
# ------------------------------------------------------------------------
# For bids, the best bid is the maximum price per timestamp.
bids = agg[agg['side'] == 'bid'][['timestamp', 'price_max']].rename(columns={'price_max': 'best_bid'})
# For asks, the best ask is the minimum price per timestamp.
asks = agg[agg['side'] == 'ask'][['timestamp', 'price_min']].rename(columns={'price_min': 'best_ask'})

# Merge the two on timestamp (only timestamps where both exist will appear)
spread = pd.merge(bids, asks, on='timestamp', how='inner')
spread['bid_ask_spread'] = spread['best_ask'] - spread['best_bid']
# Compute the mid-price as the average between best bid and best ask
spread['mid_price'] = (spread['best_bid'] + spread['best_ask']) / 2

# ------------------------------------------------------------------------
# 3. Compute Pennying Metrics
# ------------------------------------------------------------------------
# The idea is to flag orders that are "one tick" away from the best price.
# For this example, assume a tick size of 1.
tick_size = 1

# For bid orders: if the best bid minus order price equals tick_size, flag as pennying.
# For ask orders: if order price minus the best ask equals tick_size, flag as pennying.
# First, merge the best prices with the original dataframe.
# Merge best_bid into bid orders:
df_bid = df[df['side'] == 'bid']
df_bid = pd.merge(df_bid, bids, on='timestamp', how='left')
df_bid['penny_flag'] = np.where((df_bid['best_bid'] - df_bid['price']) == tick_size, 1, 0)

# Merge best_ask into ask orders:
df_ask = df[df['side'] == 'ask']
df_ask = pd.merge(df_ask, asks, on='timestamp', how='left')
df_ask['penny_flag'] = np.where((df_ask['price'] - df_ask['best_ask']) == tick_size, 1, 0)

# Recombine into one dataframe
df_with_penny = pd.concat([df_bid, df_ask], ignore_index=True)

# Now, compute the number and ratio of penny orders per timestamp and side:
penny_agg = df_with_penny.groupby(['timestamp', 'side']).agg({
    'penny_flag': ['sum', 'count']
}).reset_index()
penny_agg.columns = ['timestamp', 'side', 'penny_count', 'total_orders']
penny_agg['penny_ratio'] = penny_agg['penny_count'] / penny_agg['total_orders']

# ------------------------------------------------------------------------
# 4. Merge and Create the Final Aggregated Feature Set
# ------------------------------------------------------------------------
# Merge the basic aggregated metrics (agg) with the penny metrics and then with the spread info.
# First, merge agg and penny_agg on timestamp and side.
agg = pd.merge(agg, penny_agg, on=['timestamp', 'side'], how='left')

# Pivot the aggregated features so that bid and ask features become columns.
bid_agg = agg[agg['side'] == 'bid'].copy().reset_index(drop=True)
ask_agg = agg[agg['side'] == 'ask'].copy().reset_index(drop=True)

# Select and rename key bid features.
bid_agg_final = bid_agg[['timestamp', 'order_count', 'price_mean', 'price_std', 'price_max', 'volume_sum', 'penny_count', 'penny_ratio']]
bid_agg_final = bid_agg_final.rename(columns={
    'order_count': 'bid_order_count',
    'price_mean': 'bid_price_mean',
    'price_std': 'bid_price_std',
    'price_max': 'best_bid',  # best bid is max price in bid side.
    'volume_sum': 'bid_volume_sum',
    'penny_count': 'bid_penny_count',
    'penny_ratio': 'bid_penny_ratio'
})

# Similarly, select key ask features.
ask_agg_final = ask_agg[['timestamp', 'order_count', 'price_mean', 'price_std', 'price_min', 'volume_sum', 'penny_count', 'penny_ratio']]
ask_agg_final = ask_agg_final.rename(columns={
    'order_count': 'ask_order_count',
    'price_mean': 'ask_price_mean',
    'price_std': 'ask_price_std',
    'price_min': 'best_ask',  # best ask is min price in ask side.
    'volume_sum': 'ask_volume_sum',
    'penny_count': 'ask_penny_count',
    'penny_ratio': 'ask_penny_ratio'
})

# Merge bid and ask aggregated features on timestamp.
features = pd.merge(bid_agg_final, ask_agg_final, on='timestamp', how='inner')

# Merge in the spread info.
features = pd.merge(features, spread[['timestamp', 'bid_ask_spread', 'mid_price']], on='timestamp', how='inner')

# Reorder columns for clarity.
features = features[['timestamp', 'best_bid', 'best_ask', 'bid_ask_spread', 'mid_price',
                     'bid_order_count', 'bid_volume_sum', 'bid_price_mean', 'bid_price_std', 
                     'bid_penny_count', 'bid_penny_ratio',
                     'ask_order_count', 'ask_volume_sum', 'ask_price_mean', 'ask_price_std', 
                     'ask_penny_count', 'ask_penny_ratio']]

# Show a preview of the final aggregated features.
print("Aggregated Features for Bot Detection:")
print(features.head())

# ------------------------------------------------------------------------
# 5. Save the Aggregated Features
# ------------------------------------------------------------------------
features.to_csv("aggregated_order_features.csv", index=False)
df = pd.read_csv("aggregated_order_features.csv")

# Print a preview to inspect the features.
print("Aggregated Features:")
print(df.head())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Drop non-numeric columns (e.g., the timestamp) so we have only numeric features.
# If you want to preserve the timestamp, you can merge it back later.
features = df.drop(columns=["timestamp"]).copy()

# Standardize the features before clustering.
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Define number of clusters (adjust k based on your hypothesis)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(features_scaled)
df["cluster"] = clusters

# Use PCA to reduce dimensions to 2D for visualization.
pca = PCA(n_components=2, random_state=42)
pca_features = pca.fit_transform(features_scaled)
df["pca_one"] = pca_features[:, 0]
df["pca_two"] = pca_features[:, 1]

# Plot the clusters.
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df["pca_one"], df["pca_two"], c=df["cluster"], cmap='viridis', alpha=0.6)
plt.title("K-Means Clustering of Aggregated Order Features")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label="Cluster")
plt.show()

# Optionally, print out the cluster distribution.
cluster_counts = df["cluster"].value_counts().sort_index()
print("Cluster Distribution:")
print(cluster_counts)

# Save the DataFrame with the cluster labels for further analysis if needed.
df.to_csv("clustered_aggregated_features.csv", index=False)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data with timestamp, mid_price, cluster columns
df = pd.read_csv("clustered_aggregated_features.csv")

# (Optional) If your timestamp is a raw numeric (like 1e6 scale) and you want it to be interpreted as time, 
# convert it or leave as-is. For raw numeric, no conversion is needed. 
# If it's a datetime string, you could do something like:
# df["timestamp"] = pd.to_datetime(df["timestamp"])

# Create the plot
plt.figure(figsize=(12, 6))

# Seaborn scatterplot
sns.scatterplot(
    data=df,
    x="timestamp",
    y="mid_price",
    hue="cluster",
    style="cluster",  # Optional, assigns different markers per cluster
    palette="tab10",  # Adjust color palette if you want more distinct colors
    alpha=0.8         # Slight transparency
)

plt.title("Bot Behavior (Clusters) Over Time with Mid Price")
plt.xlabel("Timestamp")
plt.ylabel("Mid Price")
plt.legend(title="Bot Cluster")  # Ensure the legend title is descriptive

plt.show()

