
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def detect_parabolic_midprice_windows(file_paths, product='SQUID_INK', window_size=300, step=50, r2_threshold=0.80):
    # Load and combine data
    dfs = [pd.read_csv(fp, sep=';') for fp in file_paths]
    df = pd.concat(dfs, ignore_index=True)

    # Filter and compute mid-price
    df = df[df['product'] == product].copy()
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['mid_price'] = (df['ask_price_1'] + df['bid_price_1']) / 2

    results = []

    for start in range(0, len(df) - window_size, step):
        end = start + window_size
        window = df.iloc[start:end]
        if window['mid_price'].isna().any():
            continue

        t = np.arange(len(window))
        X = np.vstack([t**2, t, np.ones(len(t))]).T
        y = window['mid_price'].values

        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        if r2 >= r2_threshold:
            results.append({
                'start_timestamp': window.iloc[0]['timestamp'],
                'end_timestamp': window.iloc[-1]['timestamp'],
                'a': model.coef_[0],
                'b': model.coef_[1],
                'c': model.intercept_,
                'r_squared': r2
            })

            # Plot the actual vs fitted values
            plt.figure(figsize=(10, 4))
            plt.plot(t, y, label="Actual Mid-Price", linewidth=2)
            plt.plot(t, y_pred, label="Fitted Parabola", linestyle="--")
            plt.title(f"Parabolic Window (R²={r2:.3f}) | Time: {window.iloc[0]['timestamp']} to {window.iloc[-1]['timestamp']}")
            plt.xlabel("Ticks in Window")
            plt.ylabel("Mid Price")
            plt.legend()
            plt.grid(True)
            plt.show()

    return pd.DataFrame(results)

# Example usage
if __name__ == '__main__':
    files = ['datasets/prices_round_1_day_-2.csv', 'datasets/prices_round_1_day_-1.csv', 'datasets/prices_round_1_day_0.csv']
    parabolas = detect_parabolic_midprice_windows(files)
    print("Detected Parabolic Windows:")
    print(parabolas.to_string(index=False))
