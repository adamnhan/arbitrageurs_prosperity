from datamodel import Order, TradingState, OrderDepth
from typing import List, Dict
import numpy as np
import pandas as pd

class Trader:
    def __init__(self):
        self.mid_price_histories: Dict[str, List[float]] = {}  # Separate mid-price histories per product
        self.product_config = {
            "KELP": {
                "spread_threshold": 2,
                "order_size": 3,
                "downtrend_window": 5,
                "position_limit": 20
            },
            "RAINFOREST_RESIN": {
                "spread_threshold": 5,
                "order_size": 5,
                "downtrend_window": 10,
                "position_limit": 20
            }
        }

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        traderData = ""

        for product, order_depth in state.order_depths.items():
            config = self.product_config.get(product, {})
            spread_thresh = config.get("spread_threshold", 2)
            order_size = config.get("order_size", 3)
            window_size = config.get("downtrend_window", 5)
            position_limit = config.get("position_limit", 20)

            orders: List[Order] = []
            position = state.position.get(product, 0)

            # Get best bid/ask and spread
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2

                # Store and maintain mid-price history per product
                self.mid_price_histories.setdefault(product, []).append(mid_price)
                history = self.mid_price_histories[product]
                if len(history) > window_size:
                    history.pop(0)

                # Compute moving average for downtrend check
                if len(history) >= window_size:
                    short_term_avg = np.mean(history[-window_size:])
                    downtrend = mid_price < short_term_avg
                else:
                    downtrend = False  # Not enough data yet

                # Only trade if not in a downtrend and spread is wide enough
                if not downtrend and spread >= spread_thresh:
                    bid_price = best_bid + (spread * 0.25)
                    ask_price = best_ask - (spread * 0.25)

                    if position + order_size <= position_limit:
                        orders.append(Order(product, int(bid_price), order_size))
                    if position - order_size >= -position_limit:
                        orders.append(Order(product, int(ask_price), -order_size))

            result[product] = orders

        return result, conversions, traderData