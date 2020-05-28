#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from logbook import Logger
from collections import defaultdict
from copy import copy

from six import iteritems
import pandas as pd

from zipline.assets import Equity, Future, Option, Asset
from .blotter import Blotter
from zipline.extensions import register
from zipline.finance.order import Order
from zipline.finance.slippage import (
    DEFAULT_FUTURE_VOLUME_SLIPPAGE_BAR_LIMIT,
    VolatilityVolumeShare,
    FixedBasisPointsSlippage,
    CoverTheSpread,
)
from zipline.finance.commission import (
    DEFAULT_PER_CONTRACT_COST,
    DEFAULT_PER_OPTION_CONTRACT_COST,
    FUTURE_EXCHANGE_FEES_BY_SYMBOL,
    PerContract,
    PerShare,
    PerOptionContract,
)
from zipline.utils.input_validation import expect_types
from zipline.finance.execution import MarketOrder
from zipline.finance.transaction import create_transaction

log = Logger("Blotter")
warning_logger = Logger("AlgoWarning")


@register(Blotter, "default")
class SimulationBlotter(Blotter):
    def __init__(
        self,
        equity_slippage=None,
        future_slippage=None,
        option_slippage=None,
        equity_commission=None,
        future_commission=None,
        option_commission=None,
        cancel_policy=None,
    ):
        super(SimulationBlotter, self).__init__(cancel_policy=cancel_policy)

        # these orders are aggregated by asset
        self.open_orders = defaultdict(list)

        # keep a dict of orders by their own id
        self.orders = {}

        # holding orders that have come in since the last event.
        self.new_orders = []

        self.max_shares = int(1e11)

        self.slippage_models = {
            Equity: equity_slippage or FixedBasisPointsSlippage(),
            Future: future_slippage
            or VolatilityVolumeShare(
                volume_limit=DEFAULT_FUTURE_VOLUME_SLIPPAGE_BAR_LIMIT
            ),
            Option: option_slippage or CoverTheSpread(),
        }
        self.commission_models = {
            Equity: equity_commission or PerShare(),
            Future: future_commission
            or PerContract(
                cost=DEFAULT_PER_CONTRACT_COST,
                exchange_fee=FUTURE_EXCHANGE_FEES_BY_SYMBOL,
            ),
            Option: option_commission
            or PerOptionContract(
                cost=DEFAULT_PER_OPTION_CONTRACT_COST, exchange_fee=0.01
            ),
        }

    def __repr__(self):
        return """
{class_name}(
    slippage_models={slippage_models},
    commission_models={commission_models},
    open_orders={open_orders},
    orders={orders},
    new_orders={new_orders},
    current_dt={current_dt})
""".strip().format(
            class_name=self.__class__.__name__,
            slippage_models=self.slippage_models,
            commission_models=self.commission_models,
            open_orders=self.open_orders,
            orders=self.orders,
            new_orders=self.new_orders,
            current_dt=self.current_dt,
        )

    @expect_types(asset=Asset)
    def order(self, asset, amount, style, order_id=None):
        """Place an order.

        Parameters
        ----------
        asset : zipline.assets.Asset
            The asset that this order is for.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        style : zipline.finance.execution.ExecutionStyle
            The execution style for the order.
        order_id : str, optional
            The unique identifier for this order.

        Returns
        -------
        order_id : str or None
            The unique identifier for this order, or None if no order was
            placed.

        Notes
        -----
        amount > 0 :: Buy/Cover
        amount < 0 :: Sell/Short
        Market order:    order(asset, amount)
        Limit order:     order(asset, amount, style=LimitOrder(limit_price))
        Stop order:      order(asset, amount, style=StopOrder(stop_price))
        StopLimit order: order(asset, amount, style=StopLimitOrder(limit_price,
                               stop_price))
        """
        # something could be done with amount to further divide
        # between buy by share count OR buy shares up to a dollar amount
        # numeric == share count  AND  "$dollar.cents" == cost amount

        if amount == 0:
            # Don't bother placing orders for 0 shares.
            return None
        elif amount > self.max_shares:
            # Arbitrary limit of 100 billion (US) shares will never be
            # exceeded except by a buggy algorithm.
            raise OverflowError("Can't order more than %d shares" % self.max_shares)

        is_buy = amount > 0
        order = Order(
            dt=self.current_dt,
            asset=asset,
            amount=amount,
            stop=style.get_stop_price(is_buy),
            limit=style.get_limit_price(is_buy),
            id=order_id,
        )

        self.open_orders[order.asset].append(order)
        self.orders[order.id] = order
        self.new_orders.append(order)

        return order.id

    def cancel(self, order_id, relay_status=True):
        if order_id not in self.orders:
            return

        cur_order = self.orders[order_id]

        if cur_order.open:
            order_list = self.open_orders[cur_order.asset]
            if cur_order in order_list:
                order_list.remove(cur_order)

            if cur_order in self.new_orders:
                self.new_orders.remove(cur_order)
            cur_order.cancel()
            cur_order.dt = self.current_dt

            if relay_status:
                # we want this order's new status to be relayed out
                # along with newly placed orders.
                self.new_orders.append(cur_order)

    def cancel_all_orders_for_asset(self, asset, warn=False, relay_status=True):
        """
        Cancel all open orders for a given asset.
        """
        # (sadly) open_orders is a defaultdict, so this will always succeed.
        orders = self.open_orders[asset]

        # We're making a copy here because `cancel` mutates the list of open
        # orders in place.  The right thing to do here would be to make
        # self.open_orders no longer a defaultdict.  If we do that, then we
        # should just remove the orders once here and be done with the matter.
        for order in orders[:]:
            self.cancel(order.id, relay_status)
            if warn:
                # Message appropriately depending on whether there's
                # been a partial fill or not.
                if order.filled > 0:
                    warning_logger.warn(
                        "Your order for {order_amt} shares of "
                        "{order_sym} has been partially filled. "
                        "{order_filled} shares were successfully "
                        "purchased. {order_failed} shares were not "
                        "filled by the end of day and "
                        "were canceled.".format(
                            order_amt=order.amount,
                            order_sym=order.asset.symbol,
                            order_filled=order.filled,
                            order_failed=order.amount - order.filled,
                        )
                    )
                elif order.filled < 0:
                    warning_logger.warn(
                        "Your order for {order_amt} shares of "
                        "{order_sym} has been partially filled. "
                        "{order_filled} shares were successfully "
                        "sold. {order_failed} shares were not "
                        "filled by the end of day and "
                        "were canceled.".format(
                            order_amt=order.amount,
                            order_sym=order.asset.symbol,
                            order_filled=-1 * order.filled,
                            order_failed=-1 * (order.amount - order.filled),
                        )
                    )
                else:
                    warning_logger.warn(
                        "Your order for {order_amt} shares of "
                        "{order_sym} failed to fill by the end of day "
                        "and was canceled.".format(
                            order_amt=order.amount, order_sym=order.asset.symbol
                        )
                    )

        assert not orders
        del self.open_orders[asset]

    def execute_cancel_policy(self, event):
        if self.cancel_policy.should_cancel(event):
            warn = self.cancel_policy.warn_on_cancel
            for asset in copy(self.open_orders):
                self.cancel_all_orders_for_asset(asset, warn, relay_status=False)

    def reject(self, order_id, reason=""):
        """
        Mark the given order as 'rejected', which is functionally similar to
        cancelled. The distinction is that rejections are involuntary (and
        usually include a message from a broker indicating why the order was
        rejected) while cancels are typically user-driven.
        """
        if order_id not in self.orders:
            return

        cur_order = self.orders[order_id]

        order_list = self.open_orders[cur_order.asset]
        if cur_order in order_list:
            order_list.remove(cur_order)

        if cur_order in self.new_orders:
            self.new_orders.remove(cur_order)
        cur_order.reject(reason=reason)
        cur_order.dt = self.current_dt
        # we want this order's new status to be relayed out
        # along with newly placed orders.
        self.new_orders.append(cur_order)

    def hold(self, order_id, reason=""):
        """
        Mark the order with order_id as 'held'. Held is functionally similar
        to 'open'. When a fill (full or partial) arrives, the status
        will automatically change back to open/filled as necessary.
        """
        if order_id not in self.orders:
            return

        cur_order = self.orders[order_id]
        if cur_order.open:
            if cur_order in self.new_orders:
                self.new_orders.remove(cur_order)
            cur_order.hold(reason=reason)
            cur_order.dt = self.current_dt
            # we want this order's new status to be relayed out
            # along with newly placed orders.
            self.new_orders.append(cur_order)

    def process_splits(self, splits):
        """
        Processes a list of splits by modifying any open orders as needed.

        Parameters
        ----------
        splits: list
            A list of splits.  Each split is a tuple of (asset, ratio).

        Returns
        -------
        None
        """
        for asset, ratio in splits:
            if asset not in self.open_orders:
                continue

            orders_to_modify = self.open_orders[asset]
            for order in orders_to_modify:
                order.handle_split(ratio)

    def get_transactions(self, bar_data, data_portal=None, user_id=None):
        """
        Creates a list of transactions based on the current open orders,
        slippage model, and commission model.

        Parameters
        ----------
        user_id
        data_portal
        bar_data: zipline._protocol.BarData

        Notes
        -----
        This method book-keeps the blotter's open_orders dictionary, so that
         it is accurate by the time we're done processing open orders.

        Returns
        -------
        transactions_list: List
            transactions_list: list of transactions resulting from the current
            open orders.  If there were no open orders, an empty list is
            returned.

        commissions_list: List
            commissions_list: list of commissions resulting from filling the
            open orders.  A commission is an object with "asset" and "cost"
            parameters.

        closed_orders: List
            closed_orders: list of all the orders that have filled.
        """

        closed_orders = []
        transactions = []
        commissions = []

        if self.open_orders:
            for asset, asset_orders in iteritems(self.open_orders):
                slippage = self.slippage_models[type(asset)]

                for order, txn in slippage.simulate(bar_data, asset, asset_orders):
                    commission = self.commission_models[type(asset)]
                    additional_commission = commission.calculate(order, txn)

                    if additional_commission > 0:
                        commissions.append(
                            {
                                "asset": order.asset,
                                "order": order,
                                "cost": additional_commission,
                            }
                        )

                    order.filled += txn.amount
                    order.commission += additional_commission

                    order.dt = txn.dt

                    transactions.append(txn)

                    if not order.open:
                        closed_orders.append(order)

        return transactions, commissions, closed_orders

    def prune_orders(self, closed_orders):
        """
        Removes all given orders from the blotter's open_orders list.

        Parameters
        ----------
        closed_orders: iterable of orders that are closed.

        Returns
        -------
        None
        """
        # remove all closed orders from our open_orders dict
        for order in closed_orders:
            asset = order.asset
            asset_orders = self.open_orders[asset]
            try:
                asset_orders.remove(order)
            except ValueError:
                continue

        # now clear out the assets from our open_orders dict that have
        # zero open orders
        for asset in list(self.open_orders.keys()):
            if len(self.open_orders[asset]) == 0:
                del self.open_orders[asset]


def simulate(loaded_transactions, current_dt, orders_for_asset):

    for order, loaded_transaction in zip(orders_for_asset, loaded_transactions):

        if order.open_amount == 0:
            continue

        txn = None

        # TODO: Update this with the field that comes out of the db
        execution_volume = order.amount
        execution_price = loaded_transaction.TradePrice
        commission = loaded_transaction.Commission

        if execution_price is not None:
            txn = create_transaction(
                order, current_dt, execution_price, execution_volume
            )

        if txn:
            yield order, txn, commission


def _query_transactions(trade_datetime):

    # mock this for now, this will be a sqlalchemy db query
    path = "/Users/jason/Desktop/zipline/zipline/resources/transactions.csv"
    transactions = pd.read_csv(path, parse_dates=["TradeDatetime"]).set_index(
        "TradeDatetime"
    )
    transactions.index = pd.to_datetime(transactions.index, utc=True)
    transactions.sort_index(inplace=True)
    try:
        txn = transactions.loc[pd.Timestamp(trade_datetime.date(), tz="UTC")]
        txn = pd.DataFrame([txn]) if isinstance(txn, pd.Series) else txn
        return [x for x in txn.itertuples()]
    except KeyError:
        return []


def _check_entered_price_in_range(asset, execution_price, data_portal, current_dt):
    low = data_portal.get_spot_value(asset, "low", current_dt, "daily")
    high = data_portal.get_spot_value(asset, "high", current_dt, "daily")

    if execution_price < low or execution_price > high:
        raise ValueError(
            f"Execution price {execution_price} is outside range {low} -" f" {high}"
        )


@register(Blotter, "tradeblotterapp")
class TradeBlotterAppBlotter(SimulationBlotter):
    def __init__(self):
        super(TradeBlotterAppBlotter, self).__init__()

    def get_transactions(self, bar_data, data_portal=None):
        """
        Creates a list of transactions based on the current open orders,
        slippage model, and commission model.

        Parameters
        ----------
        bar_data: zipline._protocol.BarData

        Notes
        -----
        This method book-keeps the blotter's open_orders dictionary, so that
         it is accurate by the time we're done processing open orders.

        Returns
        -------
        transactions_list: List
            transactions_list: list of transactions resulting from the current
            open orders.  If there were no open orders, an empty list is
            returned.

        commissions_list: List
            commissions_list: list of commissions resulting from filling the
            open orders.  A commission is an object with "asset" and "cost"
            parameters.

        closed_orders: List
            closed_orders: list of all the orders that have filled.
        """

        _instrument_lookup = {
            "stock": data_portal.asset_finder.lookup_symbol,
            "etf": data_portal.asset_finder.lookup_symbol,
            "option": data_portal.asset_finder.lookup_option_symbol,
        }

        closed_orders = []
        transactions = []
        commissions = []

        loaded_transactions = defaultdict(list)

        # query the transaction table for transactions that occurred on current_dt
        # loop through and create orders which will populate self.open_orders
        todays_transactions = _query_transactions(self.current_dt)
        if todays_transactions:
            for transaction in todays_transactions:

                # TODO: Update this with the field that comes out of the db
                lookup = _instrument_lookup[transaction._1]
                asset = lookup(transaction.Symbol, self.current_dt)

                # _check_entered_price_in_range(
                #     asset, transaction.TradePrice, data_portal, self.current_dt
                # )

                loaded_transactions[asset].append(transaction)

                self.order(asset, transaction.Quantity, MarketOrder())

        if self.open_orders:
            for asset, asset_orders in iteritems(self.open_orders):

                # loop through the order and the transaction from the db
                for order, txn, commission in simulate(
                    loaded_transactions[asset], self.current_dt, asset_orders
                ):
                    # this comes from the transaction on the database. zipline expects
                    # commission values to be greater than 0
                    additional_commission = abs(commission)

                    if additional_commission > 0:
                        commissions.append(
                            {
                                "asset": order.asset,
                                "order": order,
                                "cost": additional_commission,
                            }
                        )

                    order.filled += txn.amount
                    order.commission += additional_commission

                    order.dt = txn.dt

                    transactions.append(txn)

                    if not order.open:
                        closed_orders.append(order)

        return transactions, commissions, closed_orders
