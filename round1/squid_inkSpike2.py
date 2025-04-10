import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def detect_parabolic_windows_multi(file_paths, product='SQUID_INK', window_sizes=[300, 700, 1000], step=50, r2_threshold=0.80):
    dfs = [pd.read_csv(fp, sep=';') for fp in file_paths]
    df = pd.concat(dfs, ignore_index=True)

    df = df[df['product'] == product].copy()
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['mid_price'] = (df['ask_price_1'] + df['bid_price_1']) / 2

    all_results = []

    for window_size in window_sizes:
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
                result = {
                    'window_size': window_size,
                    'start_timestamp': window.iloc[0]['timestamp'],
                    'end_timestamp': window.iloc[-1]['timestamp'],
                    'a': model.coef_[0],
                    'b': model.coef_[1],
                    'c': model.intercept_,
                    'r_squared': r2
                }
                all_results.append(result)

                # Plot the actual vs fitted values
                # plt.figure(figsize=(10, 4))
                # plt.plot(t, y, label="Actual Mid-Price", linewidth=2)
                # plt.plot(t, y_pred, label="Fitted Parabola", linestyle="--")
                # plt.title(f"Window={window_size} | R²={r2:.3f} | Time: {window.iloc[0]['timestamp']} to {window.iloc[-1]['timestamp']}")
                # plt.xlabel("Ticks in Window")
                # plt.ylabel("Mid Price")
                # plt.legend()
                # plt.grid(True)
                # plt.show()

    return pd.DataFrame(all_results)

# Example usage
if __name__ == '__main__':
    files = ['datasets/prices_round_1_day_-2.csv', 'datasets/prices_round_1_day_-1.csv', 'datasets/prices_round_1_day_0.csv']
    result_df = detect_parabolic_windows_multi(files)
    result_df.to_csv("parabolic_windows.csv", index=False)
    print("Detected Multi-Window Parabolic Fits:")
    print(result_df.to_string(index=False))