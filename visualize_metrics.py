import pandas as pd
import matplotlib.pyplot as plt

# Load the intraday metrics
kelp_df = pd.read_csv("datasets/kelp_intraday_metrics.csv")
squid_df = pd.read_csv("datasets/squid_intraday_metrics.csv")

# Set plotting style
plt.style.use("ggplot")

# --- KELP Plots ---
days = sorted(kelp_df["day_label"].unique())
for day in days:
    daily_data = kelp_df[kelp_df["day_label"] == day]

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"KELP Intraday Metrics — Day {day}", fontsize=16)

    axs[0].plot(daily_data["timestamp"], daily_data["mid_price"], label="Mid Price")
    axs[0].set_ylabel("Mid Price")
    axs[0].legend()

    axs[1].plot(daily_data["timestamp"], daily_data["bid_ask_spread"], color="orange", label="Bid-Ask Spread")
    axs[1].set_ylabel("Spread")
    axs[1].legend()

    axs[2].plot(daily_data["timestamp"], daily_data["rolling_volatility"], color="green", label="Rolling Volatility")
    axs[2].set_ylabel("Volatility")
    axs[2].legend()

    axs[3].plot(daily_data["timestamp"], daily_data["momentum"], color="purple", label="Momentum")
    axs[3].set_xlabel("Timestamp")
    axs[3].set_ylabel("Momentum")
    axs[3].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plots/kelp_intraday_day_{day}.png")
    plt.close()

# --- SQUID INK Plots ---
squid_days = sorted(squid_df["day_label"].unique())
for day in squid_days:
    daily_data = squid_df[squid_df["day_label"] == day]

    fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(f"SQUID INK Intraday Metrics — Day {day}", fontsize=16)

    axs[0].plot(daily_data["timestamp"], daily_data["mid_price"], label="Mid Price")
    axs[0].set_ylabel("Mid Price")
    axs[0].legend()

    axs[1].plot(daily_data["timestamp"], daily_data["bid_ask_spread"], color="orange", label="Bid-Ask Spread")
    axs[1].set_ylabel("Spread")
    axs[1].legend()

    axs[2].plot(daily_data["timestamp"], daily_data["rolling_volatility"], color="green", label="Rolling Volatility")
    axs[2].set_ylabel("Volatility")
    axs[2].legend()

    axs[3].plot(daily_data["timestamp"], daily_data["momentum"], color="purple", label="Momentum")
    axs[3].set_ylabel("Momentum")
    axs[3].legend()

    axs[4].plot(daily_data["timestamp"], daily_data["step_change"], color="blue", label="Step Change")
    axs[4].set_xlabel("Timestamp")
    axs[4].set_ylabel("Step Change")
    axs[4].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plots/squid_intraday_day_{day}.png")
    plt.close()

print("Plots saved to 'plots/' directory for each day for both KELP and SQUID INK.")