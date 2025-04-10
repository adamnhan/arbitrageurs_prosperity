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
        self.position_limit = 50
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
        own_trades = state.own_trades.get(product, [])
        buy_filled = any(t.buyer == 'SUBMISSION' for t in own_trades)
        sell_filled = any(t.seller == 'SUBMISSION' for t in own_trades)
        self.position = position  # update internal state

        FAIR_PRICE = 10000
        MAX_POSITION = 30
        TRADE_VOLUME = 6

        # --- Resilient Market Taking ---
        if best_ask is not None and best_ask < FAIR_PRICE and position < MAX_POSITION:
            ask_volume = abs(state.order_depths[product].sell_orders.get(best_ask, 0))
            if ask_volume <= 30:
                logger.print(f"[{state.timestamp}] Taking BUY: ask({best_ask}) < fair({FAIR_PRICE}) with vol {ask_volume}")
                volume = min(TRADE_VOLUME, MAX_POSITION - position)
                orders.append(Order(product, best_ask, volume))

        if best_bid is not None and best_bid > FAIR_PRICE and position > -MAX_POSITION:
            bid_volume = state.order_depths[product].buy_orders.get(best_bid, 0)
            if bid_volume <= 30:
                logger.print(f"[{state.timestamp}] Taking SELL: bid({best_bid}) > fair({FAIR_PRICE}) with vol {bid_volume}")
                volume = min(TRADE_VOLUME, position + MAX_POSITION)
                orders.append(Order(product, best_bid, -volume))


        # --- Edge-Aware Market Making ---
        disregard_edge = 1
        join_edge = 2
        default_edge = 3
        book_bids = list(state.order_depths[product].buy_orders.keys())
        book_asks = list(state.order_depths[product].sell_orders.keys())
        best_book_bid = max(book_bids) if book_bids else None
        best_book_ask = min(book_asks) if book_asks else None
        bid_price = FAIR_PRICE - default_edge
        ask_price = FAIR_PRICE + default_edge
        if best_book_bid is not None and best_book_bid < FAIR_PRICE - disregard_edge:
            bid_price = best_book_bid + 1 if abs(FAIR_PRICE - best_book_bid) > join_edge else best_book_bid

        # --- Dynamic MM Size Based on Spread ---
        spread = best_ask - best_bid if best_ask and best_bid else 6
        mm_volume = max(2, min(MAX_POSITION - abs(position), spread * 2))
        if best_book_ask is not None and best_book_ask > FAIR_PRICE + disregard_edge:
            ask_price = best_book_ask - 1 if abs(best_book_ask - FAIR_PRICE) > join_edge else best_book_ask
            orders.append(Order(product, bid_price, mm_volume))
        if not (buy_filled and sell_filled):
            logger.print(f"[{state.timestamp}] Posting BID at {bid_price}")
            orders.append(Order(product, bid_price, mm_volume))

        if position > -MAX_POSITION:
            logger.print(f"[{state.timestamp}] Posting ASK at {ask_price}")
            orders.append(Order(product, ask_price, -mm_volume))
            logger.print(f"[{state.timestamp}] Posting ASK at {ask_price}")
            orders.append(Order(product, ask_price, -mm_volume))

        
        # --- Zero EV Clearing Logic ---
        CLEAR_WIDTH = 0  # can be set to 1 if you want to avoid crossing mid

        if position > 0 and best_bid is not None and best_bid >= FAIR_PRICE + CLEAR_WIDTH:
            volume = min(TRADE_VOLUME, position)
            logger.print(f"[{state.timestamp}] Clear SELL {volume} @ {best_bid}")
            orders.append(Order(product, best_bid, -volume))

        elif position < 0 and best_ask is not None and best_ask <= FAIR_PRICE - CLEAR_WIDTH:
            volume = min(TRADE_VOLUME, -position)
            logger.print(f"[{state.timestamp}] Clear BUY {volume} @ {best_ask}")
            orders.append(Order(product, best_ask, volume))



        result[product] = orders
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data