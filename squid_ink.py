from datamodel import Order, TradingState, Symbol, ProsperityEncoder, Observation, OrderDepth, Trade, Listing
from typing import Dict, List, Any
from collections import deque
import json

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
        self.max_position = 20
        self.fair_value_window = 20
        self.prices = []

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        orders = {}
        conversions = 0
        trader_data = ""
        product = "SQUID_INK"

        order_depth = state.order_depths.get(product)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            logger.flush(state, orders, conversions, trader_data)
            return orders, conversions, trader_data

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        self.prices.append(mid_price)
        if len(self.prices) > self.fair_value_window:
            self.prices.pop(0)

        fair_value = sum(self.prices) / len(self.prices)
        spread = 2
        position = state.position.get(product, 0)
        inventory_penalty = (position / self.max_position) * 0.5

        bid_price = round(fair_value - spread - inventory_penalty)
        ask_price = round(fair_value + spread - inventory_penalty)

        lot_size = 5
        orders[product] = []

        if position > -self.max_position:
            orders[product].append(Order(product, bid_price, lot_size))
        if position < self.max_position:
            orders[product].append(Order(product, ask_price, -lot_size))

        logger.print(f"[{state.timestamp}] Mid: {mid_price:.2f}, Fair Value: {fair_value:.2f}, Bid: {bid_price}, Ask: {ask_price}, Pos: {position}")
        logger.flush(state, orders, conversions, trader_data)

        return orders, conversions, trader_data
