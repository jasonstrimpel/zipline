from collections import namedtuple

import pyfolio as pf
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from logbook import Logger
from functools import partial

from zipline.api import order, option_chain, record, sid
from zipline.finance import commission, slippage


log = Logger(__name__)


Leg = namedtuple("Leg", ["option", "amount"])

SELL = 1 << 0
BUY = 1 << 1
STOP = 1 << 2
LIMIT = 1 << 3
LONG = 1 << 4
SHORT = 1 << 5
DEBIT = 1 << 6
CREDIT = 1 << 7


def check_order_triggers(
    current_price, is_debit, is_long, stop_price=None, limit_price=None
):
    """Given current market value of position, returns order triggers

    Parameters
    ----------
    current_price : float
        The current value of the position, this can be negative if credit position
    is_long : boolean
        True if the entry position is a long position, false if short
    is_debit
        True if there was a cash outflow to establish this position, false if inflow
    stop_price
        Price at which the position should trigger an exit at a loss
    limit_price
        Price at which the position should trigger an exit at a gain

    Returns
    -------
    stop_reached, limit_reached : tuple of boolean

    """
    stop_reached = False
    limit_reached = False

    # order_type = 0
    # side = 0
    # position_type = 0

    # if is_long:
    #     side |= LONG
    # else:
    #     side |= SHORT
    #
    # if is_debit:
    #     position_type |= DEBIT
    # else:
    #     position_type |= CREDIT
    #
    # if stop_price is not None:
    #     order_type |= STOP
    #
    # if limit_price is not None:
    #     order_type |= LIMIT

    # if side == LONG | DEBIT:
    #     if current_price <= stop_price:
    #         stop_reached = True
    #     if current_price >= limit_price:
    #         limit_reached = True
    # elif side == SHORT | DEBIT:
    #     if current_price <= stop_price:
    #         stop_reached = True
    #     if current_price >= limit_price:
    #         limit_reached = True
    # elif side == LONG | CREDIT:
    #     if current_price <= stop_price:
    #         stop_reached = True
    #     if current_price >= limit_price:
    #         limit_reached = True
    # elif side == SHORT | CREDIT:
    #     if current_price <= stop_price:
    #         stop_reached = True
    #     if current_price >= limit_price:
    #         limit_reached = True

    # if order_type == BUY:
    #     if current_price <= stop_price:
    #         stop_reached = True
    #     if current_price >= limit_price:
    #         limit_reached = True
    # elif order_type == SELL:
    #     if current_price >= stop_price:
    #         stop_reached = True
    #     if current_price <= limit_price:
    #         limit_reached = True

    if is_debit:
        if is_long:
            if current_price <= stop_price:
                stop_reached = True
            if current_price >= limit_price:
                limit_reached = True
        else:
            if current_price >= stop_price:
                stop_reached = True
            if current_price <= limit_price:
                limit_reached = True

    else:
        if is_long:
            if current_price >= stop_price:
                stop_reached = True
            if current_price <= limit_price:
                limit_reached = True
        else:
            if current_price <= stop_price:
                stop_reached = True
            if current_price >= limit_price:
                limit_reached = True

    return stop_reached, limit_reached


def _align_expiration_with_trading_sessions(calendar, expiration_date):
    if calendar.is_session(expiration_date):
        return expiration_date

    dt = expiration_date
    offset = 1
    while not calendar.is_session(dt):
        dt = expiration_date - BDay(offset)
        offset += 1

    return dt


def dte_session(calendar, dte, expiration_date):
    aligned_expiration_date = _align_expiration_with_trading_sessions(
        calendar, expiration_date
    )
    session_window = calendar.sessions_window(aligned_expiration_date, -dte)
    return session_window[0]


class IronCondors(object):
    """Given option chain, constructs iron condors meeting requirements

    Parameters:
        option_frame : pandas.DataFrame
            Contains a well-formed fully valued options frame
        kwargs : dict
            Keywords for the iron condor search
    Returns:
        iron_condors : pandas.DataFrame
            Matching iron_condors suitable for traindg

    Example usage:

        backtest_params = {
            # days to expiration to enter trade
            "trade_entry_dte": trade_entry_dte,
            # days to expiration to exit trade
            "trade_exit_dte": 8.0,
            # look for positions less than this moneyness (< 1.0 is otm)
            "moneyness": moneyness,
            # near strike width, the short strike
            "wing_width": 20.0,
            # long strike, how many strikes out from the short strike
            "strikes_out": 1,
            # only look at condors with this net delta plus an error
            "net_delta_constraint": 0.19,
            "delta_epsilon": 0.1,
            # quantity of [LongPut, ShortPut, ShortCall, LongCall] (all > 0)
            "qty": [1, 1, 1, 1],
            # sell means a credit spread; default position
            "side": side,
            # where are we trading
            "trade_at_mid": False,
        }

        ic = IronCondors(options_frame, **backtest_params)
        condors = ic.get_iron_condors()
        trades = ic.get_trades()

    """

    def __init__(self, options_frame, **kwargs):

        self._options_frame = options_frame

        self._trade_entry_dte = kwargs["trade_entry_dte"]
        self._trade_exit_dte = kwargs["trade_exit_dte"]
        self._moneyness = kwargs["moneyness"]
        self._wing_width = kwargs["wing_width"]
        self._strikes_out = kwargs["strikes_out"]
        self._net_delta_constraint = kwargs["net_delta_constraint"]
        self._delta_epsilon = kwargs["delta_epsilon"]

        if kwargs["side"] is "sell":
            self._qty = [a * b for a, b in zip([1, -1, -1, 1], kwargs["qty"])]
        else:
            self._qty = [a * b for a, b in zip([-1, 1, 1, -1], kwargs["qty"])]

        self._filtered_options = self._filter_options()
        self.iron_condors = pd.DataFrame()

    def _filter_options(self):
        """

        """
        options_frame = self._options_frame
        trade_entry_dte = self._trade_entry_dte
        moneyness = self._moneyness

        options = options_frame[options_frame["days_to_expiration"] == trade_entry_dte]

        return options[options["moneyness"] < moneyness]

    def _get_short_strikes(self, calls, puts):
        """

        """
        wing_width = self._wing_width
        call_strike_values = calls["strike_price"].values
        put_strike_values = puts["strike_price"].values

        # create a lower triangular matrix with the differences between the strikes
        diff = np.subtract.outer(put_strike_values, call_strike_values)

        # match our wing width between call strikes and put strikes, these
        # are our short strikes
        diff_frame = pd.DataFrame(
            diff, index=put_strike_values, columns=call_strike_values
        )
        diff_frame_wings = diff_frame[diff_frame == -wing_width].notnull()

        # melt the matrix to a 2xN matrix with the short legs in the vectors and
        # rename the columns
        melted = pd.melt(diff_frame_wings.reset_index(), id_vars=["index"])
        mask = melted["value"] == True
        result = melted.loc[mask, ["index", "variable"]]
        result.columns = ["short_put_strike", "short_call_strike"]
        result.reset_index(inplace=True, drop=True)

        return result

    def _get_long_strikes(self, short_strikes):
        """

        """
        strikes_out = self._strikes_out

        short_strikes["long_put_strike"] = short_strikes["short_put_strike"].shift(
            strikes_out
        )
        short_strikes["long_call_strike"] = short_strikes["short_call_strike"].shift(
            -strikes_out
        )
        short_strikes.dropna(inplace=True)
        short_strikes.reset_index(drop=True, inplace=True)

        return short_strikes

    def _build_iron_condor_strikes(self):
        filtered_options = self._filtered_options

        # get the calls and puts sorted by strike
        calls = (
            filtered_options[filtered_options["option_type"] == "C"]
            .sort_values(["strike_price"])
            .reset_index(drop=True)
        )
        puts = (
            filtered_options[filtered_options["option_type"] == "P"]
            .sort_values(["strike_price"])
            .reset_index(drop=True)
        )

        # get the short puts and calls with strikes at wing_width
        short_strikes = self._get_short_strikes(calls, puts)

        # get the long put and call strikes at strikes_out strikes from the shorts
        self.iron_condors = self._get_long_strikes(short_strikes)

    def _build_quantity(self):
        """ Assumes qty is entered as ordered by columns

        """
        iron_condors = self.iron_condors
        qty = self._qty
        positions = len(iron_condors)
        quantities = pd.DataFrame(
            np.tile(qty, (positions, 1)),
            columns=[
                "long_put_amount",
                "short_put_amount",
                "short_call_amount",
                "long_call_amount",
            ],
        )

        self.iron_condors = pd.concat([iron_condors, quantities], axis=1)

    def _get_condor_delta(self):
        filtered_options = self._filtered_options
        iron_condors = self.iron_condors

        # get the calls and puts sorted by strike
        calls = (
            filtered_options[filtered_options["option_type"] == "C"]
            .sort_values(["strike_price"])
            .reset_index(drop=True)
        )
        puts = (
            filtered_options[filtered_options["option_type"] == "P"]
            .sort_values(["strike_price"])
            .reset_index(drop=True)
        )

        short_call_delta = calls["delta"][
            calls["strike_price"].isin(iron_condors["short_call_strike"])
        ].reset_index()
        del short_call_delta["index"]

        long_call_delta = calls["delta"][
            calls["strike_price"].isin(iron_condors["long_call_strike"])
        ].reset_index()
        del long_call_delta["index"]

        short_put_delta = puts["delta"][
            puts["strike_price"].isin(iron_condors["short_put_strike"])
        ].reset_index()
        del short_put_delta["index"]

        long_put_delta = puts["delta"][
            puts["strike_price"].isin(iron_condors["long_put_strike"])
        ].reset_index()
        del long_put_delta["index"]

        iron_condors["net_unit_delta"] = (
            short_call_delta["delta"]
            + long_call_delta["delta"]
            + short_put_delta["delta"]
            + long_put_delta["delta"]
        )

        self.iron_condors = iron_condors

    def _get_condor_sid(self):
        filtered_options = self._filtered_options
        iron_condors = self.iron_condors

        # get the calls and puts sorted by strike
        calls = (
            filtered_options[filtered_options["option_type"] == "C"]
            .sort_values(["strike_price"])
            .reset_index(drop=True)
        )
        puts = (
            filtered_options[filtered_options["option_type"] == "P"]
            .sort_values(["strike_price"])
            .reset_index(drop=True)
        )

        short_call_sid = calls["sid"][
            calls["strike_price"].isin(iron_condors["short_call_strike"])
        ].reset_index()
        del short_call_sid["index"]

        long_call_sid = calls["sid"][
            calls["strike_price"].isin(iron_condors["long_call_strike"])
        ].reset_index()
        del long_call_sid["index"]

        short_put_sid = puts["sid"][
            puts["strike_price"].isin(iron_condors["short_put_strike"])
        ].reset_index()
        del short_put_sid["index"]

        long_put_sid = puts["sid"][
            puts["strike_price"].isin(iron_condors["long_put_strike"])
        ].reset_index()
        del long_put_sid["index"]

        iron_condors["short_call_sid"] = short_call_sid
        iron_condors["long_call_sid"] = long_call_sid
        iron_condors["short_put_sid"] = short_put_sid
        iron_condors["long_put_sid"] = long_put_sid

        self.iron_condors = iron_condors

    def _filter_delta(self, iron_condors):
        net_delta_constraint = self._net_delta_constraint
        delta_epsilon = self._delta_epsilon

        r_1 = net_delta_constraint * (1.0 + delta_epsilon)
        r_2 = net_delta_constraint * (1.0 - delta_epsilon)

        delta_upper = np.maximum(r_1, r_2)
        delta_lower = np.minimum(r_1, r_2)

        where = (iron_condors["net_unit_delta"] >= delta_lower) & (
            iron_condors["net_unit_delta"] <= delta_upper
        )

        return iron_condors[where]

    def _build_iron_condors(self):
        self._build_iron_condor_strikes()
        self._build_quantity()
        self._get_condor_delta()
        self._get_condor_sid()

        self.iron_condors = self.iron_condors

    def get_iron_condors(self):
        self._build_iron_condors()
        return self.iron_condors

    def get_trades(self):
        if self._filtered_options.empty:
            return self._filtered_options
        iron_condors = self.get_iron_condors()
        return self._filter_delta(iron_condors)


def initialize(context):
    context.root_symbol = "RUT"

    context.backtest_params = {
        # days to expiration to enter trade
        "trade_entry_dte": 80,
        # days to expiration to exit trade
        "trade_exit_dte": 8,
        # look for positions less than this moneyness (< 1.0 is otm)
        "moneyness": 1.5,
        # near strike width, the short strike
        "wing_width": 20,
        # long strike, how many strikes out from the short strike
        "strikes_out": 1,
        # only look at condors with this net delta plus an error
        "net_delta_constraint": 0.16,
        "delta_epsilon": 0.1,
        # quantity of [LongPut, ShortPut, ShortCall, LongCall] (all > 0)
        "qty": [10, 10, 10, 10],
        # sell means a credit spread; default position
        "side": "sell",
    }
    # profit taking position as a percentage of the original position value
    # if short (credit) then something like 0.15 means at 15% of the original
    # credit take the winner; if long (debit) something like 2.0 means at 200%
    # of the original debit, take the winner
    context.limit_percent = 0.0  # 0.15 or 2.0
    # trade exit at a loser as a percentage of the original position value
    # if short (credit) then something like 1.6 means at 160% of the original credit
    # take the loser; if long (debit) then something like 0.75 means at 75%
    # of the original credit, take the loser
    context.stop_percent = 4.0  # 1.60
    # list of strategy which is a complex options option
    context.complex_positions = []

    context.set_commission(
        us_options=commission.PerOptionContract(
            cost=0.0075, exchange_fee=0.50, min_trade_cost=1.0
        )
    )
    # context.set_slippage(us_options=slippage.CoverTheSpread())
    context.set_slippage(us_options=slippage.NoSlippage())


def handle_data(context, data):
    log.info(f"Trade date {data.current_session}")

    # look for existing strategies to exit first which avoids immeditely
    # exiting an entered strategy
    if context.complex_positions:
        log.info(f"Found positions to trade: {context.complex_positions}")

        for complex_position in context.complex_positions:
            log.info(f"Processing {data.current_session}")

            # check if any date to exit trade has been exceeded. using any
            # covers the case where there are uneven expirations. this is a bad
            # approach because naively takes dte without considering trading days
            # if trade_exit_dte_breached is true, exit trade (covered in if below)
            dte_session_ = partial(
                dte_session,
                context.trading_calendar,
                context.backtest_params["trade_exit_dte"],
            )
            trade_exit_dte_reached = any(
                [
                    context.datetime >= dte_session_(sid(leg.option).expiration_date)
                    for leg in complex_position
                ]
            )

            # compute the cost basis of the position which is used to decide whether
            # to exit position based on the position stop loss. execution price
            # would be better than cost_basis but this value doesn't seem to exist
            cost_basis = sum(
                [
                    context.portfolio.positions[leg.option].cost_basis
                    * np.copysign(1, context.portfolio.positions[leg.option].amount)
                    for leg in complex_position
                ]
            )

            # aggregate the current value of the position based on the last_sale_price
            # which is the mid price. mid is probably a decent estimate of the closing
            # price for atm liquid options
            current_value = sum(
                [
                    context.portfolio.positions[leg.option].last_sale_price
                    * np.copysign(1, context.portfolio.positions[leg.option].amount)
                    for leg in complex_position
                ]
            )

            # if the curent value of the position, which is negative if a credit,
            # gets closer to 0 e.g. increases e.g. spreads start to collapse, we're
            # making money. therefore if the current_value is greater than the
            # basis minus what money we want to make, we buy back the position at
            # a winner
            # if the current value of the position, which is negative if a credit,
            # goes further negative e.g. spreads continue to widen, we're losing
            # money. therefore if the current_value is less than the basis
            # plus the stop loss percent, we buy back the position at a loser
            is_debit = is_long = cost_basis > 0
            stop_reached, limit_reached = check_order_triggers(
                current_value,
                is_debit,
                is_long,
                stop_price=context.stop_percent * cost_basis,
                limit_price=context.limit_percent * cost_basis,
            )
            print(f"current_value={current_value} stop_price={context.stop_percent * cost_basis} limit_price={context.limit_percent * cost_basis}")
            if stop_reached or limit_reached or trade_exit_dte_reached:
                print("in exit ")
                [order(leg.option, -leg.amount) for leg in complex_position]
                context.complex_positions.remove(complex_position)

    chain = option_chain(context.root_symbol, data.current_session)
    ic = IronCondors(chain, **context.backtest_params)
    trades = ic.get_trades()

    if not trades.empty:

        for _, trade in trades.iterrows():
            sids = trade[
                [
                    "short_call_sid",
                    "long_call_sid",
                    "short_put_sid",
                    "long_put_sid"
                ]
            ].values
            amounts = trade[
                [
                    "short_call_amount",
                    "long_call_amount",
                    "short_put_amount",
                    "long_put_amount",
                ]
            ].values

            # add the orders
            [order(sid(s), a) for s, a in zip(sids, amounts)]

            # track the orders
            context.complex_positions.append(
                [Leg(sid(s), a) for s, a in zip(sids, amounts)]
            )


def analyze(context, perf):

    returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(perf)

    round_trip_returns = pf.round_trips.extract_round_trips(
        transactions.drop("dt", axis=1)
    )
