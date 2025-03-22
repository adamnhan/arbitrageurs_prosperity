import pandas as pd
import matplotlib.pyplot as plt


def plot_market_metrics(file_path):
    """
    Reads processed market data from a CSV file and visualizes key trading metrics for each product separately.
    """
    df = pd.read_csv(file_path)

    # Convert timestamp and mid_price to numeric types for plotting
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['mid_price'] = pd.to_numeric(df['mid_price'], errors='coerce')

    # Ensure 'product' column exists
    if 'product' not in df.columns:
        raise ValueError("CSV must contain a 'product' column to separate metrics by product.")

    # Group by product and create separate plots
    products = df['product'].unique()
    for product in products:
        product_df = df[df['product'] == product].dropna(subset=['timestamp', 'mid_price'])

        fig, axs = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
        fig.suptitle(f'{product} - Market Metrics Over Time', fontsize=16)

        axs[0].plot(product_df['timestamp'], product_df['spread'], label='Spread', color='blue')
        axs[0].set_title('Spread Over Time')
        axs[0].set_ylabel('Spread')
        axs[0].legend()
        axs[0].grid()

        axs[1].plot(product_df['timestamp'], product_df['mid_price'], label='Mid Price', color='green')
        axs[1].set_title('Mid Price Over Time')
        axs[1].set_ylabel('Mid Price')
        axs[1].legend()
        axs[1].grid()

        axs[2].plot(product_df['timestamp'], product_df['volatility'], label='Volatility', color='red')
        axs[2].set_title('Volatility Over Time')
        axs[2].set_ylabel('Volatility')
        axs[2].legend()
        axs[2].grid()

        axs[3].plot(product_df['timestamp'], product_df['total_volume'], label='Total Volume', color='purple')
        axs[3].set_title('Total Volume Over Time')
        axs[3].set_ylabel('Total Volume')
        axs[3].legend()
        axs[3].grid()

        axs[4].plot(product_df['timestamp'], product_df['total_depth'], label='Order Book Depth', color='orange')
        axs[4].set_title('Order Book Depth Over Time')
        axs[4].set_xlabel('Timestamp')
        axs[4].set_ylabel('Total Depth')
        axs[4].legend()
        axs[4].grid()

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for better spacing
        plt.show()


# Example usage:
plot_market_metrics("datasets/tutorial_data_processed.csv")
