from datamodel import Order, TradingState, Symbol, ProsperityEncoder, Observation, OrderDepth, Trade, Listing
from typing import Dict, List, Any
from collections import deque
import json
import numpy as np

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
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
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for ts in trades.values() for t in ts]

    def compress_observations(self, observations: Observation) -> list[Any]:
        return [
            observations.plainValueObservations,
            {
                p: [
                    o.bidPrice, o.askPrice, o.transportFees, o.exportTariff,
                    o.importTariff, o.sugarPrice, o.sunlightIndex
                ] for p, o in observations.conversionObservations.items()
            }
        ]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for os in orders.values() for o in os]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."


logger = Logger()

class Trader:
    def __init__(self):
        self.symbol = "SQUID_INK"
        self.window = 20
        self.edge = 1
        self.beta = 0.35
        self.win = []

    def run(self, state: TradingState):
        product = self.symbol
        orders = []
        order_depth: OrderDepth = state.order_depths[product]

        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

        if best_ask is None or best_bid is None:
            return orders, 0, ""

        mid_price = (best_ask + best_bid) / 2

        self.win.append(mid_price)
        if len(self.win) > self.window:
            self.win.pop(0)

        pos = state.position.get(product, 0)
        mm_mid = np.mean(self.win)
        ret = (mid_price - mm_mid) / mm_mid if mm_mid != 0 else 0

        # Adaptive beta based on volatility
        volatility = np.std(self.win)
        dynamic_beta = self.beta / (1 + volatility / 5)
        fair = mm_mid + mm_mid * ret * dynamic_beta

        # Inventory-aware spread padding
        spread_padding = abs(pos) * 0.1

        def clip(x):
            return max(1, 10 - abs(x))

        if best_bid < fair - self.edge:
            orders.append(Order(product, int(fair - self.edge - spread_padding),  clip(pos)))

        if best_ask > fair + self.edge:
            orders.append(Order(product, int(fair + self.edge + spread_padding), -clip(-pos)))

        conversions = 0

        # Logger debugging
        logger = Logger()
        logger.print(f"Timestamp: {state.timestamp}, Mid Price: {mid_price}, Fair: {fair}, Position: {pos}")
        logger.flush(state, {product: orders}, conversions, "")

        return orders, conversions, ""