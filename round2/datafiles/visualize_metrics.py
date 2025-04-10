import pandas as pd
import matplotlib.pyplot as plt

# Load the merged data with continuous timestamps
merged_data = pd.read_csv("datasets/merged_data.csv")

# Define the products to plot side-by-side
combined_products = ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"]
for product in ["CROISSANTS", "JAM", "DJEMBE", "PICNIC_BASKET1", "PICNIC_BASKET2"]:
    product_data = merged_data[merged_data["product"] == product]
    print(f"{product}: {len(product_data)} rows, NaNs in mid_price: {product_data['mid_price'].isna().sum()}")

# Create subplots
fig, axs = plt.subplots(len(combined_products), 1, figsize=(14, 3 * len(combined_products)), sharex=True)

# Plot each product in its own subplot
for i, product in enumerate(combined_products):
    product_data = merged_data[merged_data["product"] == product]
    axs[i].plot(product_data["timestamp"], product_data["mid_price"], label=product)
    axs[i].set_title(f"{product} Mid Price Over Time")
    axs[i].set_ylabel("Mid Price")
    axs[i].grid(True)
    axs[i].legend()

axs[-1].set_xlabel("Timestamp")
plt.tight_layout()
plt.show()
