import pyfolio as pf


def initialize(context):
    pass


def before_trading_start(context, data):
    pass


def handle_data(context, data):
    pass


def analyze(context, perf):

    returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(perf)

    round_trip_returns = pf.round_trips.extract_round_trips(
        transactions.drop("dt", axis=1)
    )
