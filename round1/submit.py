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
        # KELP
        self.kelp_limit = 50
        self.kelp_prices = deque(maxlen=15)
        self.kelp_last_price = None
        self.kelp_beta = -0.3
        self.kelp_edge = 1
        self.kelp_take = 1
        self.kelp_clear = 1
        self.kelp_order_size = 15
        self.kelp_prevent_adverse = True
        self.kelp_adverse_volume = 40

        # RESIN
        self.resin_limit = 30
        self.resin_fair = 10000
        self.resin_edge = 3
        self.resin_volume = 5
        self.resin_clear_threshold = 25
        self.resin_clear_volume = 5

        # SQUID INK
        self.squid_limit = 20
        self.squid_prices = []
        self.squid_window = 20

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        orders = {}
        conversions = 0
        trader_data = ""

        for product in state.order_depths:
            od = state.order_depths[product]
            position = state.position.get(product, 0)

            if product == "KELP":
                orders[product] = self.handle_kelp(product, od, position, state.timestamp)
            elif product == "RAINFOREST_RESIN":
                orders[product] = self.handle_resin(product, od, position, state.timestamp)
            elif product == "SQUID_INK":
                orders[product] = self.handle_squid(product, od, position, state.timestamp)

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data

    def handle_kelp(self, product, od, position, timestamp):
        orders = []
        mm_bid = max([p for p, v in od.buy_orders.items() if v >= 20], default=None)
        mm_ask = min([p for p, v in od.sell_orders.items() if abs(v) >= 20], default=None)
        if mm_bid is None or mm_ask is None:
            return orders
        mm_mid = (mm_bid + mm_ask) / 2
        self.kelp_prices.append(mm_mid)
        if len(self.kelp_prices) < self.kelp_prices.maxlen:
            return orders

        fair = mm_mid
        if self.kelp_last_price is not None:
            pred_return = (mm_mid - self.kelp_last_price) / self.kelp_last_price * self.kelp_beta
            fair = mm_mid + mm_mid * pred_return
        self.kelp_last_price = mm_mid

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        bid_vol = od.buy_orders[best_bid]
        ask_vol = abs(od.sell_orders[best_ask])

        def size(limit, pos): return min(self.kelp_order_size, limit - pos)
        logger.print(f"[{timestamp}] KELP Fair: {fair}, Pos: {position}")

        if best_ask < fair - self.kelp_take and position < self.kelp_limit:
            if not self.kelp_prevent_adverse or ask_vol <= self.kelp_adverse_volume:
                orders.append(Order(product, best_ask, size(self.kelp_limit, position)))

        if best_bid > fair + self.kelp_take and position > -self.kelp_limit:
            if not self.kelp_prevent_adverse or bid_vol <= self.kelp_adverse_volume:
                orders.append(Order(product, best_bid, -size(self.kelp_limit, -position)))

        if position > 0 and best_bid >= round(fair + self.kelp_clear):
            orders.append(Order(product, best_bid, -min(self.kelp_order_size, position)))
        elif position < 0 and best_ask <= round(fair - self.kelp_clear):
            orders.append(Order(product, best_ask, min(self.kelp_order_size, -position)))

        buy_edge = round(fair - self.kelp_edge)
        sell_edge = round(fair + self.kelp_edge)
        buy_qty = size(self.kelp_limit, position)
        sell_qty = size(self.kelp_limit, -position)
        if buy_qty > 0:
            orders.append(Order(product, buy_edge, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, sell_edge, -sell_qty))

        return orders

    def handle_resin(self, product, od, position, timestamp):
        orders = []
        bid = max(od.buy_orders) if od.buy_orders else None
        ask = min(od.sell_orders) if od.sell_orders else None
        logger.print(f"[{timestamp}] RESIN Pos: {position}, Bid: {bid}, Ask: {ask}")

        if ask and ask < self.resin_fair and position < self.resin_limit:
            orders.append(Order(product, ask, min(self.resin_volume, self.resin_limit - position)))
        if bid and bid > self.resin_fair and position > -self.resin_limit:
            orders.append(Order(product, bid, -min(self.resin_volume, position + self.resin_limit)))

        if position < self.resin_limit:
            orders.append(Order(product, self.resin_fair - self.resin_edge, self.resin_volume))
        if position > -self.resin_limit:
            orders.append(Order(product, self.resin_fair + self.resin_edge, -self.resin_volume))

        if position > self.resin_clear_threshold and bid and bid >= self.resin_fair:
            orders.append(Order(product, bid, -min(self.resin_clear_volume, position)))
        elif position < -self.resin_clear_threshold and ask and ask <= self.resin_fair:
            orders.append(Order(product, ask, min(self.resin_clear_volume, -position)))

        return orders

    def handle_squid(self, product, od, position, timestamp):
        orders = []
        if not od.buy_orders or not od.sell_orders:
            return orders

        bid = max(od.buy_orders)
        ask = min(od.sell_orders)
        mid = (bid + ask) / 2
        self.squid_prices.append(mid)
        if len(self.squid_prices) > self.squid_window:
            self.squid_prices.pop(0)

        fair = sum(self.squid_prices) / len(self.squid_prices)
        penalty = (position / self.squid_limit) * 0.5
        buy_price = round(fair - 2 - penalty)
        sell_price = round(fair + 2 - penalty)

        if position > -self.squid_limit:
            orders.append(Order(product, buy_price, 5))
        if position < self.squid_limit:
            orders.append(Order(product, sell_price, -5))

        logger.print(f"[{timestamp}] SQUID Mid: {mid}, Fair: {fair}, Pos: {position}")
        return orders
