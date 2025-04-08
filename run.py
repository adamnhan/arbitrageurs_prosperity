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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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

    def kelp_fair_value(self, order_depth: OrderDepth, trader_data: dict) -> float:
        adverse_volume = 15
        reversion_beta = -0.2

        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        if buy_orders and sell_orders:
            filtered_bid = [price for price, size in buy_orders.items() if size >= adverse_volume]
            filtered_ask = [price for price, size in sell_orders.items() if size >= adverse_volume]

            mm_bid = max(filtered_bid) if filtered_bid else None
            mm_ask = min(filtered_ask) if filtered_ask else None
            best_bid = max(buy_orders.keys())
            best_ask = min(sell_orders.keys())

            if mm_bid is not None and mm_ask is not None:
                mm_mid = (mm_bid + mm_ask) / 2
            elif best_bid is not None and best_ask is not None:
                mm_mid = (best_bid + best_ask) / 2
            else:
                mm_mid = trader_data.get("kelp_last_price", 10000)

            if "kelp_last_price" in trader_data:
                last_price = trader_data["kelp_last_price"]
                last_return = (mm_mid - last_price) / last_price
                predicted_return = last_return * reversion_beta
                fair_price = mm_mid + mm_mid * predicted_return
            else:
                fair_price = mm_mid

            trader_data["kelp_last_price"] = mm_mid
            return fair_price

        return trader_data.get("kelp_last_price", 10000)

    def __init__(self):
        self.price_history = []
        self.window_size = 15
        self.position_limit = 50
        self.order_size = 50  # aggressive sizing
        # Multipliers for dynamic threshold
        self.k1 = 0.7  # weight for volatility component
        self.k2 = 0.3  # weight for bid-ask spread component

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        symbol = "KELP"
        position = defaultdict(int, state.position)
        current_pos = position[symbol]
        order_depth = state.order_depths.get(symbol, OrderDepth())

        
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        # Market Maker Filtering
        mm_bid_prices = [price for price, size in order_depth.buy_orders.items() if size >= 20]
        mm_ask_prices = [price for price, size in order_depth.sell_orders.items() if size >= 20]

        mm_best_bid = max(mm_bid_prices) if mm_bid_prices else best_bid
        mm_best_ask = min(mm_ask_prices) if mm_ask_prices else best_ask

        mm_mid_price = (mm_best_bid + mm_best_ask) / 2 if mm_best_bid is not None and mm_best_ask is not None else None
        mid_price = mm_mid_price  # Now mid_price is replaced with market maker mid

        logger.print(f"[{state.timestamp}] MM Mid Price Used: {mid_price}")

        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2

        logger.print(f"[{state.timestamp}] Mid Price: {mid_price}")

        # Insert after `best_ask`, `best_bid`, and `mid_price` logic
        mm_bid_size = sum(order_depth.buy_orders.values())
        mm_ask_size = -sum(order_depth.sell_orders.values())
        mm_spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None

        # Thresholds can be tuned
        mm_present = (
            mm_bid_size >= 15 and
            mm_ask_size >= 15 and
            mm_spread is not None and
            2 <= mm_spread <= 5
        )

        logger.print(f"[{state.timestamp}] MM Detected: {mm_present}, Bid Size: {mm_bid_size}, Ask Size: {mm_ask_size}, Spread: {mm_spread}")


        if mid_price is not None:
            self.price_history.append(mid_price)
            if len(self.price_history) > self.window_size:
                self.price_history.pop(0)

        orders = []
        dynamic_threshold = None
        signal_strength = None
        deviation = None

        if mid_price is not None and len(self.price_history) == self.window_size:
            moving_average = sum(self.price_history) / self.window_size
            deviation = (mid_price - moving_average) / moving_average
            rolling_std = np.std(self.price_history)
            bid_ask_spread = best_ask - best_bid if best_bid is not None and best_ask is not None else 0

            dynamic_threshold = self.k1 * (rolling_std / moving_average) + self.k2 * (bid_ask_spread / moving_average)
            if mm_present:
                dynamic_threshold *= 0.8
            signal_strength = abs(deviation) - dynamic_threshold

            logger.print(
                f"[{state.timestamp}] MA: {moving_average:.2f}, "
                f"Dev: {deviation:.5f}, "
                f"Vol: {rolling_std:.5f}, "
                f"Spread: {bid_ask_spread}, "
                f"Threshold: {dynamic_threshold:.5f}, "
                f"Signal Strength: {signal_strength:.5f}"
            )

            if deviation > dynamic_threshold and current_pos > -self.position_limit:
                volume = min(self.order_size, self.position_limit + current_pos)
                if best_bid is not None and volume > 0:
                    logger.print(f"[{state.timestamp}] Placing SELL order at {best_bid} for {volume} units.")
                    orders.append(Order(symbol, best_bid, -volume))
            elif deviation < -dynamic_threshold and current_pos < self.position_limit:
                volume = min(self.order_size, self.position_limit - current_pos)
                if best_ask is not None and volume > 0:
                    logger.print(f"[{state.timestamp}] Placing BUY order at {best_ask} for {volume} units.")
                    orders.append(Order(symbol, best_ask, volume))
            else:
                if deviation > dynamic_threshold and current_pos <= -self.position_limit:
                    logger.print(f"[{state.timestamp}] Cannot SELL: Position limit reached.")
                elif deviation < -dynamic_threshold and current_pos >= self.position_limit:
                    logger.print(f"[{state.timestamp}] Cannot BUY: Position limit reached.")
                else:
                    logger.print(f"[{state.timestamp}] No trade: Signal too weak.")
        else:
            logger.print(f"[{state.timestamp}] Not enough data to calculate deviation.")

        result[symbol] = orders

        trader_data = json.dumps({
            "mid_price": mid_price,
            "moving_average": sum(self.price_history) / len(self.price_history) if self.price_history else None,
            "rolling_std": np.std(self.price_history) if self.price_history else None,
            "bid_ask_spread": (best_ask - best_bid) if best_bid is not None and best_ask is not None else None,
            "dynamic_threshold": dynamic_threshold,
            "deviation": deviation,
            "current_pos": current_pos,
            "orders": [order.__dict__ for order in orders]
        })

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data