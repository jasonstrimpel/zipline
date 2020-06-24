#
# Copyright 2013 Quantopian, Inc.
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
import os

import pandas as pd
import requests

from . import loader
from trading_calendars import get_calendar
from zipline.data import bundles
from zipline.data.data_portal import DataPortal


def get_benchmark_returns(symbol, bundle_name="quandl-eod", calendar_name="NYSE"):
    """Use the zipline data portal to return benchmark data

    Parameters
    ----------
    symbol : str
        Benchmark symbol string
    bundle_name : str
        The name of the zipline bundle to look for data in
    calendar_name : str
        The calendar that returns the benchmark prices

    Returns
    -------
    returns :

    """
    calendar = get_calendar(calendar_name)
    bundle_data = bundles.load(bundle_name)

    start_date = pd.Timestamp("1990-01-03", tz="UTC")
    end_date = pd.Timestamp("today", tz="UTC")
    bar_count = calendar.session_distance(start_date, end_date)

    portal = DataPortal(
        bundle_data.asset_finder,
        calendar,
        bundle_data.equity_daily_bar_reader.first_trading_day,
        equity_minute_reader=bundle_data.equity_minute_bar_reader,
        equity_daily_reader=bundle_data.equity_daily_bar_reader,
        adjustment_reader=bundle_data.adjustment_reader,
    )

    prices = portal.get_history_window(
        assets=[portal.asset_finder.lookup_symbol(symbol, end_date)],
        end_dt=end_date,
        bar_count=bar_count,
        frequency="1d",
        data_frequency="daily",
        field="close",
    )
    prices.columns = symbol

    #
    # api_key = os.environ.get('IEX_API_KEY')
    # if api_key is None:
    #     raise ValueError(
    #         "Please set your IEX_API_KEY environment variable and retry."
    #     )
    # r = requests.get(
    #     "https://cloud.iexapis.com/stable/stock/{}/chart/5y?token={}".format(symbol, api_key)
    # )
    #
    # if r.status_code != 200:
    #     path = loader.get_data_filepath(loader.get_benchmark_filename(symbol))
    #     df = pd.read_csv(path, names=["date", "return"])
    #     df.index = pd.DatetimeIndex(df["date"], tz="UTC")
    #     return df["return"]
    #
    # data = r.json()
    #
    # df = pd.DataFrame(data)
    #
    # df.index = pd.DatetimeIndex(df["date"])
    # df = df["close"]

    return prices.sort_index().dropna().pct_change(1).iloc[1:]
