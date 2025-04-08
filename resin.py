from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
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
        self.position = 0
        self.position_limit = 30
        self.mid_prices = []

    def get_mid_price(self, best_bid, best_ask):
        return (best_bid + best_ask) / 2

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""
        product = "RAINFOREST_RESIN"
        orders = []

        best_bid = max(state.order_depths[product].buy_orders.keys()) if state.order_depths[product].buy_orders else None
        best_ask = min(state.order_depths[product].sell_orders.keys()) if state.order_depths[product].sell_orders else None

        position = state.position.get(product, 0)
        self.position = position  # update internal state

        FAIR_PRICE = 10000
        EDGE = 3
        MAX_POSITION = 30
        TRADE_VOLUME = 5

        # --- Market Taking ---
        if best_ask is not None and best_ask < FAIR_PRICE and position < MAX_POSITION:
            logger.print(f"[{state.timestamp}] Taking BUY: ask({best_ask}) < fair({FAIR_PRICE})")
            volume = min(TRADE_VOLUME, MAX_POSITION - position)
            orders.append(Order(product, best_ask, volume))

        if best_bid is not None and best_bid > FAIR_PRICE and position > -MAX_POSITION:
            logger.print(f"[{state.timestamp}] Taking SELL: bid({best_bid}) > fair({FAIR_PRICE})")
            volume = min(TRADE_VOLUME, position + MAX_POSITION)
            orders.append(Order(product, best_bid, -volume))

        # --- Market Making ---
        bid_price = FAIR_PRICE - EDGE
        ask_price = FAIR_PRICE + EDGE

        if position < MAX_POSITION:
            logger.print(f"[{state.timestamp}] Posting BID at {bid_price}")
            orders.append(Order(product, bid_price, TRADE_VOLUME))

        if position > -MAX_POSITION:
            logger.print(f"[{state.timestamp}] Posting ASK at {ask_price}")
            orders.append(Order(product, ask_price, -TRADE_VOLUME))

        
        # --- Zero EV Clearing Logic ---
        CLEAR_THRESHOLD = 25
        CLEAR_ORDER_SIZE = 5

        if position > CLEAR_THRESHOLD and best_bid is not None and best_bid >= FAIR_PRICE:
            volume = min(CLEAR_ORDER_SIZE, position)
            logger.print(f"[{state.timestamp}] Clearing long: SELL {volume} at {best_bid}")
            orders.append(Order(product, best_bid, -volume))

        elif position < -CLEAR_THRESHOLD and best_ask is not None and best_ask <= FAIR_PRICE:
            volume = min(CLEAR_ORDER_SIZE, -position)
            logger.print(f"[{state.timestamp}] Clearing short: BUY {volume} at {best_ask}")
            orders.append(Order(product, best_ask, volume))


        result[product] = orders
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data