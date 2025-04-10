from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
from collections import deque, defaultdict
import statistics
import numpy as np
import json
import math

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json([
                self.compress_state(state, ""),
                self.compress_orders(orders),
                conversions,
                "",
                "",
            ])
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json([
                self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                self.compress_orders(orders),
                conversions,
                self.truncate(trader_data, max_item_length),
                self.truncate(self.logs, max_item_length),
            ])
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [od.buy_orders, od.sell_orders] for symbol, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for arr in trades.values() for t in arr]

    def compress_observations(self, observations: Observation) -> list[Any]:
        return [
            observations.plainValueObservations,
            {
                product: [
                    obs.bidPrice, obs.askPrice, obs.transportFees, obs.exportTariff,
                    obs.importTariff, obs.sugarPrice, obs.sunlightIndex
                ] for product, obs in observations.conversionObservations.items()
            }
        ]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."


logger = Logger()

class Trader:
    def __init__(self):
        self.symbol = "SQUID_INK"
        self.max_position = 50
        self.trade_size = 5
        self.windows = [300, 700, 1000]
        self.weights = {300: 3, 700: 0.5, 1000: 0.5}
        self.price_histories = {w: deque(maxlen=w) for w in self.windows}

        self.models = {
            300: {
                "convex": {"a": 0.000894, "b": -0.03472, "c": 1953.843, "r2": 0.951},
                "concave": {"a": -0.000512, "b": 0.21346, "c": 1960.577, "r2": 0.947}
            },
            700: {
                "convex": {"a": 0.000190, "b": 0.03471, "c": 2058.350, "r2": 0.933},
                "concave": {"a": -0.000187, "b": 0.07543, "c": 1914.702, "r2": 0.935}
            },
            1000: {
                "convex": {"a": 0.000170, "b": -0.04635, "c": 2052.262, "r2": 0.946},
                "concave": {"a": -0.000023, "b": -0.03410, "c": 2040.159, "r2": 0.873}
            }
        }

    def compute_midprice(self, od: OrderDepth) -> float:
        best_bid = max(od.buy_orders.keys(), default=None)
        best_ask = min(od.sell_orders.keys(), default=None)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None

    def fit_and_score(self, window: int, prices: deque) -> float:
        if len(prices) < window:
            return 0.0

        t = np.arange(len(prices))
        y = np.array(prices)
        current_price = y[-1]
        score = 0

        for shape in ["convex", "concave"]:
            model = self.models[window][shape]
            if model["r2"] < 0.85:
                continue

            predicted = model["a"] * t[-1]**2 + model["b"] * t[-1] + model["c"]

            if model["a"] > 0 and current_price < predicted:
                score += 1
            elif model["a"] < 0 and current_price > predicted:
                score -= 1

        return score * self.weights[window]

    def run(self, state: TradingState):
        od = state.order_depths.get(self.symbol)
        if od is None:
            return {}, 0, ""

        pos = state.position.get(self.symbol, 0)
        mid_price = self.compute_midprice(od)
        if mid_price is None:
            return {}, 0, ""

        for w in self.windows:
            self.price_histories[w].append(mid_price)

        total_score = sum(self.fit_and_score(w, self.price_histories[w]) for w in self.windows)

        orders: List[Order] = []
        best_bid = max(od.buy_orders.keys(), default=None)
        best_ask = min(od.sell_orders.keys(), default=None)

        if total_score >= 2 and pos < self.max_position and best_ask is not None:
            orders.append(Order(self.symbol, best_ask, self.trade_size))
            logger.print(f"[Score = {total_score:.2f}] BUY at {best_ask}")

        elif total_score <= -2 and pos > -self.max_position and best_bid is not None:
            orders.append(Order(self.symbol, best_bid, -self.trade_size))
            logger.print(f"[Score = {total_score:.2f}] SELL at {best_bid}")

        trader_data = json.dumps({
            "mid_price": float(mid_price),
            "position": int(pos),
            "score": float(total_score)
        })

        logger.flush(state, {self.symbol: orders}, 0, trader_data)
        return {self.symbol: orders}, 0, trader_data
