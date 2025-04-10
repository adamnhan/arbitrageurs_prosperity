import json
import pandas as pd
import matplotlib.pyplot as plt

def analyze_trade_history(trade_history_file):
    """
    Reads a JSON trade history from a text file, processes the data, 
    and provides key trading insights and visualizations.
    """
    
    # Load trade history from text file
    with open(trade_history_file, 'r') as file:
        data = file.read()
        
    # Extract JSON part
    trade_data = json.loads(data[data.index('['):])
    
    # Convert to DataFrame
    df = pd.DataFrame(trade_data)
    
    # Convert timestamp to numeric type for sorting & plotting
    df['timestamp'] = df['timestamp'].astype(int)
    
    # Identify trades made by our bot ("SUBMISSION")
    df['is_buy'] = df['buyer'] == "SUBMISSION"
    df['is_sell'] = df['seller'] == "SUBMISSION"
    
    # Calculate profit/loss per trade
    df['pnl'] = df.apply(lambda row: row['price'] * row['quantity'] * (1 if row['is_sell'] else -1), axis=1)
    
    # Aggregate P&L over time
    df['cumulative_pnl'] = df['pnl'].cumsum()
    
    # Group by timestamp to analyze trading activity
    trade_summary = df.groupby('timestamp').agg({
        'quantity': 'sum',
        'price': ['mean', 'min', 'max'],
        'pnl': 'sum'
    }).reset_index()
    trade_summary.columns = ['timestamp', 'total_quantity', 'avg_price', 'min_price', 'max_price', 'total_pnl']
    
    # Save processed trade summary to CSV
    trade_summary.to_csv("processed_trade_history.csv", index=False)
    
    # Plot cumulative P&L
    plt.figure(figsize=(10,5))
    plt.plot(df['timestamp'], df['cumulative_pnl'], label='Cumulative P&L', color='purple')
    plt.xlabel('Timestamp')
    plt.ylabel('Profit & Loss (SeaShells)')
    plt.title('Trading Performance Over Time')
    plt.legend()
    plt.grid()
    plt.show()
    
    return df, trade_summary

# Example usage:
# analyze_trade_history("trade_history.txt")
analyze_trade_history("datasets/trade_history.txt")
