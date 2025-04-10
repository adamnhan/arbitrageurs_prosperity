from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
from collections import deque, defaultdict
import statistics
import numpy as np
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
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        self.position_limit = 50
        self.price_window = deque(maxlen=15)
        self.last_price = None
        self.reversion_beta = -0.3
        self.default_edge = 1
        self.take_threshold = 0.3
        self.clear_threshold = 1
        self.base_order_size = 15
        self.prevent_adverse = True
        self.adverse_volume = 40

    def find_mm_mid(self, order_depth: OrderDepth, volume_threshold=20):
        mm_bid_prices = [p for p, v in order_depth.buy_orders.items() if v >= volume_threshold]
        mm_ask_prices = [p for p, v in order_depth.sell_orders.items() if abs(v) >= volume_threshold]

        mm_bid = max(mm_bid_prices) if mm_bid_prices else None
        mm_ask = min(mm_ask_prices) if mm_ask_prices else None

        if mm_bid is not None and mm_ask is not None:
            return (mm_bid + mm_ask) / 2
        return None

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        result = {}
        conversions = 0
        symbol = "KELP"
        trader_data = {}

        position = state.position.get(symbol, 0)
        order_depth: OrderDepth = state.order_depths.get(symbol, OrderDepth())
        orders = []

        mm_mid = self.find_mm_mid(order_depth)
        if mm_mid is not None:
            self.price_window.append(mm_mid)

            if len(self.price_window) == self.price_window.maxlen:
                base_fair = sum(self.price_window) / len(self.price_window)

                if self.last_price is not None:
                    predicted_return = (mm_mid - self.last_price) / self.last_price * self.reversion_beta
                    fair = mm_mid + mm_mid * predicted_return
                else:
                    fair = mm_mid

                self.last_price = mm_mid

                best_bid = max(order_depth.buy_orders)
                best_ask = min(order_depth.sell_orders)
                bid_volume = order_depth.buy_orders[best_bid]
                ask_volume = abs(order_depth.sell_orders[best_ask])

                logger.print(f"[{state.timestamp}] MM Mid: {mm_mid}, Fair (rev): {fair}, Pos: {position}")

                def adjusted_order_size(limit, pos):
                    return min(self.base_order_size, limit - pos)

                # TAKER
                if best_ask < fair - self.take_threshold and position < self.position_limit:
                    if not self.prevent_adverse or ask_volume <= self.adverse_volume:
                        qty = adjusted_order_size(self.position_limit, position)
                        orders.append(Order(symbol, best_ask, qty))
                        logger.print(f"TAKE BUY {qty} @ {best_ask}")

                if best_bid > fair + self.take_threshold and position > -self.position_limit:
                    if not self.prevent_adverse or bid_volume <= self.adverse_volume:
                        qty = adjusted_order_size(self.position_limit, -position)
                        orders.append(Order(symbol, best_bid, -qty))
                        logger.print(f"TAKE SELL {qty} @ {best_bid}")

                # CLEAR
                if position > 0:
                    target_ask = round(fair + self.clear_threshold)
                    if best_bid >= target_ask:
                        qty = min(self.base_order_size, position)
                        orders.append(Order(symbol, best_bid, -qty))
                        logger.print(f"CLEAR SELL {qty} @ {best_bid}")

                elif position < 0:
                    target_bid = round(fair - self.clear_threshold)
                    if best_ask <= target_bid:
                        qty = min(self.base_order_size, -position)
                        orders.append(Order(symbol, best_ask, qty))
                        logger.print(f"CLEAR BUY {qty} @ {best_ask}")

                # MARKET MAKER
                buy_edge = round(fair - self.default_edge)
                sell_edge = round(fair + self.default_edge)

                buy_qty = adjusted_order_size(self.position_limit, position)
                sell_qty = adjusted_order_size(self.position_limit, -position)

                if buy_qty > 0:
                    orders.append(Order(symbol, buy_edge, buy_qty))
                    logger.print(f"MM BID {buy_qty} @ {buy_edge}")
                if sell_qty > 0:
                    orders.append(Order(symbol, sell_edge, -sell_qty))
                    logger.print(f"MM ASK {sell_qty} @ {sell_edge}")

        result[symbol] = orders

        trader_data_json = json.dumps({
            "fair": fair if 'fair' in locals() else None,
            "orders": [order.__dict__ for order in result.get(symbol, [])]
        })

        logger.flush(state, result, conversions, trader_data_json)
        return result, conversions, trader_data_json
