import pandas as pd

def calculate_and_save_metrics(input_csv: str, output_csv: str):
    """
    Reads a CSV file of market activity data, calculates spread, volatility, volume,
    and order book depth metrics, and writes the processed data to a new CSV file.
    
    Args:
        input_csv (str): Path to the input CSV (semicolon or comma delimited).
        output_csv (str): Path to save the new CSV with calculated metrics.
    """
    # Detect delimiter
    with open(input_csv, 'r') as f:
        sample_line = f.readline()
    delimiter = ';' if ';' in sample_line else ','

    # Load the CSV with the correct delimiter
    df = pd.read_csv(input_csv, sep=delimiter)

    # Ensure numeric columns are properly typed
    df['timestamp'] = df['timestamp'].astype(int)
    df['mid_price'] = df['mid_price'].astype(float)
    df['return_pct'] = df.groupby('product')['mid_price'].pct_change() * 100

    # Metric 1: Spread
    df['spread'] = df['ask_price_1'] - df['bid_price_1']

    # Metric 2: Volatility (rolling std of mid_price over 5 ticks)
    df['volatility'] = df['mid_price'].rolling(window=5).std()

    # Metric 3: Top-of-book volume (bid_volume_1 + ask_volume_1)
    df['total_volume'] = df['bid_volume_1'].fillna(0).abs() + df['ask_volume_1'].fillna(0).abs()

    # Metric 4 & 5: Order book depth (sum of volumes across top 3 levels)
    bid_cols = ['bid_volume_1', 'bid_volume_2', 'bid_volume_3']
    ask_cols = ['ask_volume_1', 'ask_volume_2', 'ask_volume_3']
    df['bid_depth'] = df[bid_cols].fillna(0).abs().sum(axis=1)
    df['ask_depth'] = df[ask_cols].fillna(0).abs().sum(axis=1)
    df['total_depth'] = df['bid_depth'] + df['ask_depth']

    # Save the enriched data
    df.to_csv(output_csv, index=False)
    print(f"✅ Metrics calculated and saved to {output_csv}")

calculate_and_save_metrics("datasets/tutorial_data.csv", "datasets/tutorial_data_processed.csv")
