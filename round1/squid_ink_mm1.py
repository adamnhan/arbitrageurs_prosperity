import pandas as pd
import os

# Load the merged CSV file
df = pd.read_csv("datasets/merged_prices_round_1.csv")

# Filter for SQUID_INK only
squid_df = df[df["product"] == "SQUID_INK"].copy()

# Collect summary rows to output
summary_rows = []

# Loop through rows
for i, row in squid_df.iterrows():
    timestamp = row["timestamp"]
    bids = [(row.get(f"bid_price_{j}"), row.get(f"bid_volume_{j}")) for j in range(1, 4)]
    asks = [(row.get(f"ask_price_{j}"), row.get(f"ask_volume_{j}")) for j in range(1, 4)]

    print(f"Timestamp: {timestamp}")
    print("  Bids:")
    for price, vol in bids:
        print(f"    Price: {price}, Volume: {vol}")
    print("  Asks:")
    for price, vol in asks:
        print(f"    Price: {price}, Volume: {vol}")
    print("-" * 40)

    # Save to output row
    row_data = {
        "timestamp": timestamp,
        "bid_price_1": bids[0][0], "bid_volume_1": bids[0][1],
        "bid_price_2": bids[1][0], "bid_volume_2": bids[1][1],
        "bid_price_3": bids[2][0], "bid_volume_3": bids[2][1],
        "ask_price_1": asks[0][0], "ask_volume_1": asks[0][1],
        "ask_price_2": asks[1][0], "ask_volume_2": asks[1][1],
        "ask_price_3": asks[2][0], "ask_volume_3": asks[2][1],
    }
    summary_rows.append(row_data)

# Create DataFrame
summary_df = pd.DataFrame(summary_rows)

# Ensure 'datasets' folder exists
os.makedirs("datasets", exist_ok=True)

# Save to CSV
summary_df.to_csv("datasets/squid_order_book_summary.csv", index=False)
print("✅ Summary CSV saved to datasets/squid_order_book_summary.csv")
