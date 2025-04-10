
import pandas as pd

# Load and filter for KELP
df = pd.read_csv("datasets/merged_prices_round_1.csv", delimiter=",", low_memory=False)
kelp_df = df[df["product"] == "KELP"].copy()

# Select relevant columns for top 3 bid/ask levels
columns = [
    "timestamp",
    "bid_price_1", "bid_volume_1",
    "bid_price_2", "bid_volume_2",
    "bid_price_3", "bid_volume_3",
    "ask_price_1", "ask_volume_1",
    "ask_price_2", "ask_volume_2",
    "ask_price_3", "ask_volume_3"
]

# Filter and save
output_df = kelp_df[columns]
output_df.to_csv("datasets/kelp_order_book_top3.csv", index=False)
print("Saved to kelp_order_book_top3.csv")
