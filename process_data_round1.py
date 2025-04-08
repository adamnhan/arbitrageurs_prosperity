import pandas as pd

# Load CSVs with semicolon delimiter
day_0 = pd.read_csv("datasets/prices_round_1_day_0.csv", delimiter=";")
day_neg1 = pd.read_csv("datasets/prices_round_1_day_-1.csv", delimiter=";")
day_neg2 = pd.read_csv("datasets/prices_round_1_day_-2.csv", delimiter=";")

# Add day labels
day_0["day_label"] = 0
day_neg1["day_label"] = -1
day_neg2["day_label"] = -2

# Merge all into one DataFrame
merged_df = pd.concat([day_neg2, day_neg1, day_0], ignore_index=True)

# Convert relevant columns to numeric
merged_df["timestamp"] = pd.to_numeric(merged_df["timestamp"], errors="coerce")
merged_df["mid_price"] = pd.to_numeric(merged_df["mid_price"], errors="coerce")
merged_df["bid_price_1"] = pd.to_numeric(merged_df["bid_price_1"], errors="coerce")
merged_df["ask_price_1"] = pd.to_numeric(merged_df["ask_price_1"], errors="coerce")
merged_df["bid_volume_1"] = pd.to_numeric(merged_df["bid_volume_1"], errors="coerce")

# Sort for consistency
merged_df = merged_df.sort_values(by=["product", "day_label", "timestamp"]).reset_index(drop=True)
merged_df.to_csv("datasets/merged_prices_round_1.csv", index=False)

# --- Analysis for KELP ---
kelp_df = merged_df[merged_df["product"] == "KELP"].copy()

# Group by day and calculate metrics
kelp_metrics = kelp_df.groupby("day_label").agg(
    intra_day_volatility=("mid_price", lambda x: x.max() - x.min()),
    avg_bid_ask_spread=("mid_price", lambda x: (kelp_df["ask_price_1"] - kelp_df["bid_price_1"]).mean()),
    open_price=("mid_price", lambda x: x.iloc[0]),
    close_price=("mid_price", lambda x: x.iloc[-1]),
    avg_bid_volume=("bid_volume_1", "mean")
)

# Calculate daily returns and volume spike
kelp_metrics["daily_return"] = kelp_metrics["close_price"].pct_change()
kelp_metrics["volume_spike"] = kelp_metrics["avg_bid_volume"] / kelp_metrics["avg_bid_volume"].rolling(2).mean()

# Save kelp metrics
kelp_metrics.to_csv("datasets/kelp_metrics.csv")

# --- Intra-day KELP Metrics Across Timestamps ---

# Calculate bid-ask spread
kelp_df["bid_ask_spread"] = kelp_df["ask_price_1"] - kelp_df["bid_price_1"]

# Rolling volatility over past 5 timestamps per day
kelp_df["rolling_volatility"] = kelp_df.groupby("day_label")["mid_price"].transform(lambda x: x.rolling(window=5, min_periods=1).std())

# Momentum: price change over past 3 timestamps
kelp_df["momentum"] = kelp_df.groupby("day_label")["mid_price"].transform(lambda x: x.diff(periods=3))

# Select and save relevant columns
kelp_intraday_metrics = kelp_df[[
    "day_label", "timestamp", "mid_price", "bid_price_1", "ask_price_1",
    "bid_ask_spread", "rolling_volatility", "momentum"
]].copy()

kelp_intraday_metrics.to_csv("datasets/kelp_intraday_metrics.csv", index=False)

# --- Intra-day SQUID INK Metrics ---
squid_df = merged_df[merged_df["product"] == "SQUID_INK"].copy()

# Calculate bid-ask spread
squid_df["bid_ask_spread"] = squid_df["ask_price_1"] - squid_df["bid_price_1"]

# Rolling volatility (oscillation signal)
squid_df["rolling_volatility"] = squid_df.groupby("day_label")["mid_price"].transform(lambda x: x.rolling(window=5, min_periods=1).std())

# Momentum (price change across 3 timestamps)
squid_df["momentum"] = squid_df.groupby("day_label")["mid_price"].transform(lambda x: x.diff(periods=3))

# Step change detection: large price jumps
squid_df["step_change"] = squid_df.groupby("day_label")["mid_price"].transform(lambda x: x.diff().abs())

# Select and save relevant columns
squid_intraday_metrics = squid_df[[
    "day_label", "timestamp", "mid_price", "bid_price_1", "ask_price_1",
    "bid_ask_spread", "rolling_volatility", "momentum", "step_change"
]].copy()

squid_intraday_metrics.to_csv("datasets/squid_intraday_metrics.csv", index=False)