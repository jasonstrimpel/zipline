"""
Module for building a complete daily dataset from ivolatility's options data.
"""
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from click import progressbar
from logbook import Logger
from six import iteritems
from six.moves.urllib.parse import urlencode
from sqlalchemy import create_engine

from zipline.assets import AssetFinder
from zipline.data.bcolz_daily_bars import BcolzDailyBarReader
from zipline.data.bundles.core import most_recent_data
from zipline.data.data_portal import DataPortal
from zipline.data.loader import load_market_data
from zipline.data.treasuries import get_interpolated_rate
from zipline.finance import valuation
from . import core as bundles


log = Logger(__name__)

ONE_MEGABYTE = 1024 * 1024
IVOLATILITY_DATA_URL = "https://www.dropbox.com/s/rim7neyn5sdzno9/ibm.large.csv?"

QUANDL_PATH = most_recent_data("quandl", pd.Timestamp("now"))


def format_metadata_url(api_key):
    """ Build the query URL for Quandl WIKI Prices metadata.
    """
    query_params = [("api_key", api_key), ("dl", 1)]

    return IVOLATILITY_DATA_URL + urlencode(query_params)


def load_data_table(file, index_col, show_progress=False):
    """ Load data table from CSV file provided by ivolatility.
    """
    if show_progress:
        log.info("Parsing raw data.")

    data_table = pd.read_csv(
        file,
        parse_dates=["date", "expiration"],
        index_col=index_col,
        usecols=[
            "date",
            "symbol",
            "exchange",
            "adjusted stock close price",
            "option symbol",
            "expiration",
            "strike",
            "call/put",
            "style",
            "ask",
            "bid",
            "volume",
            "open interest",
            "unadjusted stock price",
        ],
    )
    data_table.rename(
        columns={
            "symbol": "root_symbol",
            "adjusted stock close price": "adjusted_underlying_close",
            "option symbol": "symbol",
            "strike": "strike_price",
            "expiration": "expiration_date",
            "call/put": "option_type",
            "open interest": "open_interest",
            "unadjusted stock price": "unadjusted_underlying_close",
        },
        inplace=True,
        copy=False,
    )
    return data_table


def fetch_data_table(api_key, show_progress, retries):
    """ Fetch price data table from ivolatility
    """
    for _ in range(retries):
        try:
            if show_progress:
                log.info("Downloading metadata.")

            table_url = format_metadata_url(api_key)
            if show_progress:
                raw_file = download_with_progress(
                    table_url,
                    chunk_size=ONE_MEGABYTE,
                    label="Downloading option price table from ivolatility",
                )
            else:
                raw_file = download_without_progress(table_url)

            return load_data_table(
                file=raw_file, index_col=None, show_progress=show_progress
            )

        except Exception:
            log.exception("Exception raised reading ivolatility data. Retrying.")

    else:
        raise ValueError(
            "Failed to download ivolatility data after %d attempts." % (retries)
        )


def gen_root_symbols(data, show_progress):
    if show_progress:
        log.info("Generating asset root symbols.")

    data = data.groupby(by="root_symbol").agg({"exchange": "first"})
    data.reset_index(inplace=True)
    return data


def gen_asset_metadata(data, show_progress):
    if show_progress:
        log.info("Generating asset metadata.")

    data = data.groupby(by="occ_symbol").agg(
        {
            "symbol": "first",
            "root_symbol": "first",
            "date": [np.min, np.max],
            "exchange": "first",
            "expiration_date": "first",
            "strike_price": "first",
            "option_type": "first",
            "style": "first",
        }
    )
    data.reset_index(inplace=True)
    data["start_date"] = data.date.amin
    data["end_date"] = data.date.amax
    del data["date"]
    data.columns = data.columns.get_level_values(0)

    data["asset_name"] = ""
    data["tick_size"] = 0.01
    data["multiplier"] = 100.0
    data["first_traded"] = data["start_date"]
    data["auto_close_date"] = data["expiration_date"].values + pd.Timedelta(days=1)
    return data


def _gen_symbols(data, show_progress):
    if show_progress:
        log.info("Generating OCC symbols.")

    data["symbol"] = [x.replace(" ", "") for x in data.symbol.values]

    root_symbol_fmt = [
        "{:6}".format(x.upper()).replace(" ", "") for x in data.root_symbol.values
    ]

    expiration_fmt = [
        pd.Timestamp(x).strftime("%y%m%d")
        for x in data.expiration_date.values.astype("datetime64[D]")
    ]

    option_type_fmt = [x.upper() for x in data.option_type.values]

    strike_fmt = [
        "{:09.3f}".format(float(x)).replace(".", "") for x in data.strike_price.values
    ]

    occ_format = lambda x: f"{x[0]}{x[1]}{x[2]}{x[3]}"
    mapped = map(
        occ_format, zip(root_symbol_fmt, expiration_fmt, option_type_fmt, strike_fmt)
    )

    data["occ_symbol"] = list(mapped)
    return data


def _get_price_metadata(data, show_progress):
    if show_progress:
        log.info("Generating mid, spread, moneyness and days to expiration.")

    bid = data.bid.values
    ask = data.ask.values

    bid[np.isnan(bid)] = 0.0
    ask[np.isnan(ask)] = 0.0

    data["mid"] = (ask + bid) / 2.0
    data["spread"] = ask - bid

    # create the close price because it's used everywhere
    data["close"] = data.mid

    data["moneyness"] = np.nan

    calls = data[data.option_type == "C"]
    puts = data[data.option_type == "P"]

    data.loc[data.option_type == "C", "moneyness"] = (
        calls.strike_price.values / calls.unadjusted_underlying_close.values
    )
    data.loc[data.option_type == "P", "moneyness"] = (
        puts.unadjusted_underlying_close.values / puts.strike_price.values
    )
    data["days_to_expiration"] = (
        data.expiration_date.values - data.date.values
    ).astype("timedelta64[D]")
    return data


def _gen_interest_rates(data, treasury_curves, show_progress):
    if show_progress:
        log.info("Generating interpolated interest rates.")

    for name, group in data.groupby(["date", "days_to_expiration"]):
        date, days_to_expiration = name
        rates = treasury_curves.ix[date].values
        data.loc[group.index, "interest_rate"] = get_interpolated_rate(
            rates, days_to_expiration.days
        )

    return data


def _gen_annualized_volatility(data, calendar, show_progress):
    if show_progress:
        log.info("Generating underlying annualized volatility.")

    asset_finder = AssetFinder(
        create_engine(f"sqlite:///{QUANDL_PATH}/assets-7.sqlite")
    )
    bars = 30

    first_trading_day = calendar.sessions_window(
        pd.Timestamp(data.date.min(), tz="UTC"), -bars
    )[0]
    data_portal = DataPortal(
        asset_finder=asset_finder,
        equity_daily_reader=BcolzDailyBarReader(f"{QUANDL_PATH}/daily_equities.bcolz"),
        trading_calendar=calendar,
        first_trading_day=first_trading_day,
    )

    for name, group in data.groupby(["root_symbol", "date"]):
        root_symbol, date = name
        equity = asset_finder.lookup_symbol(
            root_symbol, as_of_date=pd.Timestamp(date, tz="UTC")
        )
        data.loc[
            group.index, "statistical_volatility"
        ] = data_portal.get_simple_transform(
            asset=equity,
            transform_name="annualized_volatility",
            dt=date,
            data_frequency="daily",
            field="close",
            bars=bars,
        )

    return data


def _valuation_models(option_type, style):
    option_type = "call" if option_type.upper() == "C" else "put"
    model = (
        f"binomial_american_{option_type}_value"
        if style.upper() == "A"
        else f"black_scholes_{option_type}_value"
    )
    return getattr(valuation, model)


def _greeks_models(option_type, style):
    option_type = "call" if option_type.upper() == "C" else "put"
    delta = (
        f"binomial_american_{option_type}_delta"
        if style.upper() == "A"
        else f"black_scholes_{option_type}_delta"
    )
    gamma = (
        f"binomial_american_{option_type}_gamma"
        if style.upper() == "A"
        else f"black_scholes_{option_type}_gamma"
    )
    vega = (
        f"binomial_american_{option_type}_vega"
        if style.upper() == "A"
        else f"black_scholes_{option_type}_vega"
    )
    theta = (
        f"binomial_american_{option_type}_theta"
        if style.upper() == "A"
        else f"black_scholes_{option_type}_theta"
    )
    rho = (
        f"binomial_american_{option_type}_rho"
        if style.upper() == "A"
        else f"black_scholes_{option_type}_rho"
    )
    return {
        "delta": getattr(valuation, delta),
        "gamma": getattr(valuation, gamma),
        "vega": getattr(valuation, vega),
        "theta": getattr(valuation, theta),
        "rho": getattr(valuation, rho),
    }


def _implied_volatility_models(option_type, style):
    option_type = "call" if option_type.upper() == "C" else "put"
    model = (
        f"binomial_american_{option_type}_implied_volatility_brent"
        if style.upper() == "A"
        else f"black_scholes_{option_type}_implied_volatility_brent"
    )
    return getattr(valuation, model)


def _gen_option_valuation(data, calendar, show_progress):
    if show_progress:
        log.info("Generating option valuation.")

    asset_finder = AssetFinder(
        create_engine(f"sqlite:///{QUANDL_PATH}/assets-7.sqlite")
    )
    first_trading_day = pd.Timestamp(data.date.min(), tz="UTC")
    data_portal = DataPortal(
        asset_finder=asset_finder,
        equity_daily_reader=BcolzDailyBarReader(f"{QUANDL_PATH}/daily_equities.bcolz"),
        trading_calendar=calendar,
        first_trading_day=first_trading_day,
    )

    val_fcn_v = np.vectorize(_valuation_models)
    greek_fcn_v = np.vectorize(_greeks_models)

    for name, group in data.groupby(["root_symbol", "date"]):
        root_symbol, date = name
        equity = asset_finder.lookup_symbol(
            root_symbol, as_of_date=pd.Timestamp(date, tz="UTC")
        )
        s_ = data_portal.get_spot_value(
            equity, "close", pd.Timestamp(date, tz="UTC"), "daily"
        )

        val_fcns = val_fcn_v(group.option_type.values, group["style"].values)
        greek_fcns = greek_fcn_v(group.option_type.values, group["style"].values)

        k = group.strike_price.values
        s = np.full(k.shape, s_)
        r = group.interest_rate.values
        t = group.days_to_expiration.astype("timedelta64[D]").values / 365.0
        vol = group.statistical_volatility.values

        data.loc[group.index, "option_value"] = [
            fn(s_, k_, r_, t_, vol_)
            for fn, s_, k_, r_, t_, vol_ in zip(val_fcns, s, k, r, t, vol)
        ]
        data.loc[group.index, "delta"] = [
            fn["delta"](s_, k_, r_, t_, vol_)
            for fn, s_, k_, r_, t_, vol_ in zip(greek_fcns, s, k, r, t, vol)
        ]
        data.loc[group.index, "gamma"] = [
            fn["gamma"](s_, k_, r_, t_, vol_)
            for fn, s_, k_, r_, t_, vol_ in zip(greek_fcns, s, k, r, t, vol)
        ]
        data.loc[group.index, "vega"] = [
            fn["vega"](s_, k_, r_, t_, vol_)
            for fn, s_, k_, r_, t_, vol_ in zip(greek_fcns, s, k, r, t, vol)
        ]
        data.loc[group.index, "theta"] = [
            fn["theta"](s_, k_, r_, t_, vol_)
            for fn, s_, k_, r_, t_, vol_ in zip(greek_fcns, s, k, r, t, vol)
        ]
        data.loc[group.index, "rho"] = [
            fn["rho"](s_, k_, r_, t_, vol_)
            for fn, s_, k_, r_, t_, vol_ in zip(greek_fcns, s, k, r, t, vol)
        ]

    return data


def _gen_implied_volatility(data, calendar, show_progress):
    if show_progress:
        log.info("Generating option implied volatility.")

    asset_finder = AssetFinder(
        create_engine(f"sqlite:///{QUANDL_PATH}/assets-7.sqlite")
    )
    first_trading_day = pd.Timestamp(data.date.min(), tz="UTC")
    data_portal = DataPortal(
        asset_finder=asset_finder,
        equity_daily_reader=BcolzDailyBarReader(f"{QUANDL_PATH}/daily_equities.bcolz"),
        trading_calendar=calendar,
        first_trading_day=first_trading_day,
    )

    iv_fcn_v = np.vectorize(_implied_volatility_models)

    for name, group in data.groupby(["root_symbol", "date"]):
        root_symbol, date = name
        equity = asset_finder.lookup_symbol(
            root_symbol, as_of_date=pd.Timestamp(date, tz="UTC")
        )
        s_ = data_portal.get_spot_value(
            equity, "close", pd.Timestamp(date, tz="UTC"), "daily"
        )

        iv_fcns = iv_fcn_v(group.option_type.values, group["style"].values)

        k = group.strike_price.values
        s = np.full(k.shape, s_)
        r = group.interest_rate.values
        t = group.days_to_expiration.astype("timedelta64[D]").values / 365.0
        mid_price = group.mid.values

        data.loc[group.index, "implied_volatility"] = [
            fn(s_, k_, r_, t_, mid_price_)
            for fn, s_, k_, r_, t_, mid_price_ in zip(iv_fcns, s, k, r, t, mid_price)
        ]

    return data


def gen_valuation_metadata(data, treasury_curves, calendar, show_progress):
    data = _gen_symbols(data, show_progress)
    data = _get_price_metadata(data, show_progress)
    data = _gen_interest_rates(data, treasury_curves, show_progress)
    data = _gen_annualized_volatility(data, calendar, show_progress)
    data = _gen_option_valuation(data, calendar, show_progress)
    data = _gen_implied_volatility(data, calendar, show_progress)

    return data


def parse_pricing_and_vol(data, sessions, symbol_map):
    for asset_id, occ_symbol in iteritems(symbol_map):
        asset_data = (
            data.xs(occ_symbol, level=1).reindex(sessions.tz_localize(None)).fillna(0.0)
        )
        yield asset_id, asset_data


@bundles.register("ivolatility")
def ivolatility_bundle(
    environ,
    asset_db_writer,
    minute_bar_writer,
    daily_bar_writer,
    daily_chain_writer,
    adjustment_writer,
    calendar,
    start_session,
    end_session,
    cache,
    show_progress,
    output_dir,
):
    """
    ivolatility_bundle builds a daily dataset using ivolatility's dataset.
    """
    api_key = environ.get("IVOLATILITY_API_KEY")
    if api_key is None:
        raise ValueError(
            "Please set your IVOLATILITY_API_KEY environment variable and retry."
        )
    raw_data = fetch_data_table(
        api_key, show_progress, environ.get("IVOLATILITY_DOWNLOAD_ATTEMPTS", 5)
    )
    _, treasury_curves = load_market_data(
        trading_day=calendar.day, trading_days=calendar.all_sessions, environ=environ
    )
    raw_data = gen_valuation_metadata(
        raw_data, treasury_curves, calendar, show_progress
    )
    asset_metadata = gen_asset_metadata(raw_data, show_progress)

    root_symbols = gen_root_symbols(
        raw_data[["root_symbol", "exchange"]], show_progress
    )
    asset_db_writer.write(options=asset_metadata, root_symbols=root_symbols)

    symbol_map = asset_metadata.occ_symbol
    sessions = calendar.sessions_in_range(start_session, end_session)

    raw_data.set_index(["date", "occ_symbol"], inplace=True)

    daily_chain_writer.write(
        parse_pricing_and_vol(raw_data, sessions, symbol_map),
        show_progress=show_progress,
    )


def download_with_progress(url, chunk_size, **progress_kwargs):
    """
    Download streaming data from a URL, printing progress information to the
    terminal.

    Parameters
    ----------
    url : str
        A URL that can be understood by ``requests.get``.
    chunk_size : int
        Number of bytes to read at a time from requests.
    **progress_kwargs
        Forwarded to click.progressbar.

    Returns
    -------
    data : BytesIO
        A BytesIO containing the downloaded data.
    """
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    total_size = int(resp.headers["content-length"])
    data = BytesIO()
    with progressbar(length=total_size, **progress_kwargs) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            data.write(chunk)
            pbar.update(len(chunk))

    data.seek(0)
    return data


def download_without_progress(url):
    """
    Download data from a URL, returning a BytesIO containing the loaded data.

    Parameters
    ----------
    url : str
        A URL that can be understood by ``requests.get``.

    Returns
    -------
    data : BytesIO
        A BytesIO containing the downloaded data.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    return BytesIO(resp.content)
