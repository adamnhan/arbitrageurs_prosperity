from datamodel import Order, OrderDepth, ProsperityEncoder, TradingState, Symbol, Observation, Listing, Trade
from typing import List, Dict, Any
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
        self.component_map = {
            "PICNIC_BASKET1": {"CROISSANTS": 6, "JAM": 3, "DJEMBE": 1},
            "PICNIC_BASKET2": {"CROISSANTS": 4, "JAM": 2}
        }
        self.position_limit = {
            "CROISSANTS": 250,
            "JAM": 350,
            "DJEMBE": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100
        }

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            return (best_bid + best_ask) / 2
        return None

    def can_fully_unwind(self, component_quantities: Dict[str, int], order_depths: Dict[str, OrderDepth]) -> bool:
        for product, qty_needed in component_quantities.items():
            if product not in order_depths or not order_depths[product].buy_orders:
                return False
            best_bid_vol = order_depths[product].buy_orders[max(order_depths[product].buy_orders)]
            if best_bid_vol < qty_needed:
                return False
        return True

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        order_depths = state.order_depths
        result = {}
        conversions = 0
        trader_data_json = json.dumps({})

        # Track how long we've been holding each component
        if not hasattr(self, "hold_tracker"):
            self.hold_tracker = {}

        for comp in ["CROISSANTS", "JAM", "DJEMBE"]:
            pos = state.position.get(comp, 0)
            if pos > 0:
                if comp not in self.hold_tracker:
                    self.hold_tracker[comp] = state.timestamp
                else:
                    held_time = state.timestamp - self.hold_tracker[comp]
                    logger.print(f"[HOLDING] {comp}: {pos} units for {held_time} ticks")
            else:
                self.hold_tracker.pop(comp, None)

        for basket, components in self.component_map.items():
            if basket not in order_depths:
                continue

            basket_depth = order_depths[basket]
            if not basket_depth.sell_orders:
                continue

            best_basket_ask = min(basket_depth.sell_orders)
            basket_vol = basket_depth.sell_orders[best_basket_ask]

            sellable_components = {}
            unsellable_components = []
            for comp, qty in components.items():
                if comp in order_depths and order_depths[comp].buy_orders:
                    best_bid = max(order_depths[comp].buy_orders)
                    best_bid_vol = order_depths[comp].buy_orders[best_bid]
                    if best_bid_vol >= qty:
                        sellable_components[comp] = (best_bid, qty)
                    else:
                        unsellable_components.append(comp)
                else:
                    unsellable_components.append(comp)

            if len(sellable_components) == 0:
                continue

            implied_value = 0
            for comp, qty in components.items():
                if comp in order_depths and order_depths[comp].buy_orders:
                    best_bid = max(order_depths[comp].buy_orders)
                    implied_value += best_bid * qty

            profit = implied_value - best_basket_ask
            if profit > 0:
                logger.print(f"[ARBITRAGE] {basket}: profit = {profit}, ask = {best_basket_ask}, implied = {implied_value}")
                logger.print(f"  Sellable: {list(sellable_components.keys())}, Stuck: {unsellable_components}")

                result[basket] = [Order(basket, best_basket_ask, 1)]
                for comp, (price, qty) in sellable_components.items():
                    result.setdefault(comp, []).append(Order(comp, price, -qty))

        logger.flush(state, result, conversions, trader_data_json)
        return result, conversions, trader_data_json

