from functools import partial
import warnings

from bcolz import carray, ctable
import logbook
import numpy as np
from numpy import array, full, iinfo, nan
import pandas as pd
from pandas import DatetimeIndex, NaT, read_csv, to_datetime, Timestamp
from six import iteritems, viewkeys
from toolz import compose
from trading_calendars import get_calendar

from zipline.data.session_bars import CurrencyAwareSessionBarReader
from zipline.data.bar_reader import NoDataAfterDate, NoDataBeforeDate, NoDataOnDate
from zipline.utils.functional import apply
from zipline.utils.input_validation import expect_element
from zipline.utils.numpy_utils import iNaT, float64_dtype, uint32_dtype, int64_dtype
from zipline.utils.memoize import lazyval
from zipline.utils.cli import maybe_show_progress
from ._options import _compute_row_slices, _read_bcolz_data

logger = logbook.Logger("UsOptionsPricing")

# windsorize these columns, coerce to unsigned int32
PRICING = frozenset(
    [
        "adjusted_underlying_close",
        "ask",
        "bid",
        "close",
        "implied_volatility",
        "interest_rate",
        "mid",
        "moneyness",
        "option_value",
        "spread",
        "statistical_volatility",
        "strike_price",
        "unadjusted_underlying_close",
    ]
)

# windsorize these columns, coerce to signed int32
GREEKS = frozenset(["delta", "gamma", "rho", "theta", "vega"])

# these are the columns that are stored to disk
OPTION_PRICING_BCOLZ_COLUMNS = (
    "adjusted_underlying_close",
    "ask",
    "bid",
    "close",
    "delta",
    "gamma",
    "interest_rate",
    "implied_volatility",
    "mid",
    "moneyness",
    "option_value",
    "rho",
    "spread",
    "statistical_volatility",
    "strike_price",
    "theta",
    "unadjusted_underlying_close",
    "vega",
    # fields not explicity in the PRICING or GREEKS sets
    "open_interest",
    "volume",
    "days_to_expiration",
    "id",
    "day",
)

UINT32_MAX = iinfo(np.uint32).max


def check_uint32_safe(value, colname):
    if value >= UINT32_MAX:
        raise ValueError("Value %s from column '%s' is too large" % (value, colname))


@expect_element(invalid_data_behavior={"warn", "raise", "ignore"})
def winsorise_uint32(df, invalid_data_behavior, column, *columns):
    """Drops any record where a value would not fit into a uint32.
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to winsorise.
    invalid_data_behavior : {'warn', 'raise', 'ignore'}
        What to do when data is outside the bounds of a uint32.
    *columns : iterable[str]
        The names of the columns to check.
    Returns
    -------
    truncated : pd.DataFrame
        ``df`` with values that do not fit into a uint32 zeroed out.
    """
    # remove datetime columns to avoid masking mixed dtypes errors
    df = df.select_dtypes(exclude=["object", "datetime64", "timedelta64"])
    columns = list((column,) + columns)
    mask = df[columns] > UINT32_MAX

    if invalid_data_behavior != "ignore":
        mask |= df[columns].isnull()
    else:
        # we are not going to generate a warning or error for this so just use
        # nan_to_num
        df[columns] = np.nan_to_num(df[columns])

    mv = mask.values
    if mv.any():
        if invalid_data_behavior == "raise":
            raise ValueError(
                "%d values out of bounds for uint32: %r"
                % (mv.sum(), df[mask.any(axis=1)])
            )
        if invalid_data_behavior == "warn":
            warnings.warn(
                "Ignoring %d values because they are out of bounds for"
                " uint32: %r" % (mv.sum(), df[mask.any(axis=1)]),
                stacklevel=3,  # one extra frame for `expect_element`
            )

    df[mask] = 0
    return df


class BcolzDailyChainWriter(object):
    """
    Class capable of writing daily chain price data to disk in a format that can
    be read efficiently by BcolzDailyChainReader.
    Parameters
    ----------
    filename : str
        The location at which we should write our output.
    calendar : zipline.utils.calendar.trading_calendar
        Calendar to use to compute asset calendar offsets.
    start_session: pd.Timestamp
        Midnight UTC session label.
    end_session: pd.Timestamp
        Midnight UTC session label.
    See Also
    --------
    zipline.data.us_equity_pricing.BcolzDailyBarReader
    """

    _csv_dtypes = {
        "adjusted_underlying_close": float64_dtype,
        "strike": float64_dtype,
        "ask": float64_dtype,
        "bid": float64_dtype,
        "close": float64_dtype,
        "mid": float64_dtype,
        "spread": float64_dtype,
        "moneyness": float64_dtype,
        "volume": uint32_dtype,
        "open_interest": uint32_dtype,
        "unadjusted_underlying_close": float64_dtype,
        "interest_rate": float64_dtype,
        "statistical_volatility": float64_dtype,
        "option_value": float64_dtype,
        "delta": float64_dtype,
        "gamma": float64_dtype,
        "vega": float64_dtype,
        "theta": float64_dtype,
        "rho": float64_dtype,
    }

    def __init__(self, filename, calendar, start_session, end_session):
        self._filename = filename

        if start_session != end_session:
            if not calendar.is_session(start_session):
                raise ValueError("Start session %s is invalid!" % start_session)
            if not calendar.is_session(end_session):
                raise ValueError("End session %s is invalid!" % end_session)

        self._start_session = start_session
        self._end_session = end_session

        self._calendar = calendar

    @property
    def progress_bar_message(self):
        return "Merging daily options files:"

    def progress_bar_item_show_func(self, value):
        return value if value is None else str(value[0])

    def write(
        self, data, assets=None, show_progress=False, invalid_data_behavior="warn"
    ):
        """
        Parameters
        ----------
        data : iterable[tuple[int, pandas.DataFrame or bcolz.ctable]]
            The data chunks to write. Each chunk should be a tuple of sid
            and the data for that asset.
        assets : set[int], optional
            The assets that should be in ``data``. If this is provided
            we will check ``data`` against the assets and provide better
            progress information.
        show_progress : bool, optional
            Whether or not to show a progress bar while writing.
        invalid_data_behavior : {'warn', 'raise', 'ignore'}, optional
            What to do when data is encountered that is outside the range of
            a uint32.
        Returns
        -------
        table : bcolz.ctable
            The newly-written table.
        """
        ctx = maybe_show_progress(
            ((sid, self.to_ctable(df, invalid_data_behavior)) for sid, df in data),
            show_progress=show_progress,
            item_show_func=self.progress_bar_item_show_func,
            label=self.progress_bar_message,
            length=len(assets) if assets is not None else None,
        )
        with ctx as it:
            return self._write_internal(it, assets)

    def write_csvs(self, asset_map, show_progress=False, invalid_data_behavior="warn"):
        """Read CSVs as DataFrames from our asset map.

        Parameters
        ----------
        asset_map : dict[int -> str]
            A mapping from asset id to file path with the CSV data for that
            asset
        show_progress : bool
            Whether or not to show a progress bar while writing.
        invalid_data_behavior : {'warn', 'raise', 'ignore'}
            What to do when data is encountered that is outside the range of
            a uint32.
        """
        read = partial(
            read_csv, parse_dates=["day"], index_col="day", dtype=self._csv_dtypes
        )
        return self.write(
            ((asset, read(path)) for asset, path in iteritems(asset_map)),
            assets=viewkeys(asset_map),
            show_progress=show_progress,
            invalid_data_behavior=invalid_data_behavior,
        )

    def _write_internal(self, iterator, assets):
        """
        Internal implementation of write.
        `iterator` should be an iterator yielding pairs of (asset, ctable).
        """
        total_rows = 0
        first_row = {}
        last_row = {}
        calendar_offset = {}

        # Maps column name -> output carray adding int64 for the greeks
        columns = {
            k: carray(
                array([], dtype=(uint32_dtype if k not in GREEKS else int64_dtype))
            )
            for k in OPTION_PRICING_BCOLZ_COLUMNS
        }

        earliest_date = None
        sessions = self._calendar.sessions_in_range(
            self._start_session, self._end_session
        )

        if assets is not None:

            @apply
            def iterator(iterator=iterator, assets=set(assets)):
                for asset_id, table in iterator:
                    if asset_id not in assets:
                        raise ValueError("unknown asset id %r" % asset_id)
                    yield asset_id, table

        for asset_id, table in iterator:

            logger.info(f"Writing asset id {asset_id} to disk")

            nrows = len(table)
            for column_name in columns:
                if column_name == "id":
                    # We know what the content of this column is, so don't
                    # bother reading it.
                    columns["id"].append(full((nrows,), asset_id, dtype="uint32"))
                    continue

                columns[column_name].append(table[column_name])

            if earliest_date is None:
                earliest_date = table["day"][0]
            else:
                earliest_date = min(earliest_date, table["day"][0])

            # Bcolz doesn't support ints as keys in `attrs`, so convert
            # assets to strings for use as attr keys.
            asset_key = str(asset_id)

            # Calculate the index into the array of the first and last row
            # for this asset. This allows us to efficiently load single
            # assets when querying the data back out of the table.
            first_row[asset_key] = total_rows
            last_row[asset_key] = total_rows + nrows - 1
            total_rows += nrows

            table_day_to_session = compose(
                self._calendar.minute_to_session_label,
                partial(Timestamp, unit="s", tz="UTC"),
            )
            asset_first_day = table_day_to_session(table["day"][0])
            asset_last_day = table_day_to_session(table["day"][-1])

            asset_sessions = sessions[
                sessions.slice_indexer(asset_first_day, asset_last_day)
            ]
            assert len(table) == len(asset_sessions), (
                "Got {} rows for daily bars table with first day={}, last "
                "day={}, expected {} rows.\n"
                "Missing sessions: {}\n"
                "Extra sessions: {}".format(
                    len(table),
                    asset_first_day.date(),
                    asset_last_day.date(),
                    len(asset_sessions),
                    asset_sessions.difference(
                        to_datetime(np.array(table["day"]), unit="s", utc=True)
                    ).tolist(),
                    to_datetime(np.array(table["day"]), unit="s", utc=True)
                    .difference(asset_sessions)
                    .tolist(),
                )
            )

            # Calculate the number of trading days between the first date
            # in the stored data and the first date of **this** asset. This
            # offset used for output alignment by the reader.
            calendar_offset[asset_key] = sessions.get_loc(asset_first_day)

        logger.info("Writing complete table to disk")
        # This writes the table to disk.
        full_table = ctable(
            columns=[columns[colname] for colname in OPTION_PRICING_BCOLZ_COLUMNS],
            names=OPTION_PRICING_BCOLZ_COLUMNS,
            rootdir=self._filename,
            mode="w",
        )

        full_table.attrs["first_trading_day"] = (
            earliest_date if earliest_date is not None else iNaT
        )

        full_table.attrs["first_row"] = first_row
        full_table.attrs["last_row"] = last_row
        full_table.attrs["calendar_offset"] = calendar_offset
        full_table.attrs["calendar_name"] = self._calendar.name
        full_table.attrs["start_session_ns"] = self._start_session.value
        full_table.attrs["end_session_ns"] = self._end_session.value
        full_table.flush()
        return full_table

    @expect_element(invalid_data_behavior={"warn", "raise", "ignore"})
    def to_ctable(self, raw_data, invalid_data_behavior):
        if isinstance(raw_data, ctable):
            # we already have a ctable so do nothing
            return raw_data

        # windorise the pricing fields plus volume and open interest
        winsorise_uint32(raw_data, invalid_data_behavior, "volume", *PRICING)
        winsorise_uint32(raw_data, invalid_data_behavior, "open_interest", *PRICING)

        # process the pricing fields and greeks separatly (greeks signed)
        processed_pricing = (raw_data[list(PRICING)] * 1000).round().astype("uint32")
        processed_greeks = (raw_data[list(GREEKS)] * 1000).round().astype("int64")
        processed = pd.concat([processed_pricing, processed_greeks], axis=1)

        # process the dates
        dates = raw_data.index.values.astype("datetime64[s]")
        days_to_expiration = raw_data.days_to_expiration.values.astype("timedelta64[D]")
        check_uint32_safe(dates.max().view(np.int64), "day")
        check_uint32_safe(days_to_expiration.max().view(np.int64), "days_to_expiration")
        processed["day"] = dates.astype("uint32")
        processed["days_to_expiration"] = days_to_expiration.astype("uint32")
        processed["volume"] = raw_data.volume.astype("uint32")
        processed["open_interest"] = raw_data.open_interest.astype("uint32")

        return ctable.fromdataframe(processed)


class BcolzDailyChainReader(CurrencyAwareSessionBarReader):
    """
    Reader for raw pricing data written by BcolzDailyOHLCVWriter.
    Parameters
    ----------
    table : bcolz.ctable
        The ctable contaning the pricing data, with attrs corresponding to the
        Attributes list below.
    read_all_threshold : int
        The number of equities at which; below, the data is read by reading a
        slice from the carray per asset.  above, the data is read by pulling
        all of the data for all assets into memory and then indexing into that
        array for each day and asset pair.  Used to tune performance of reads
        when using a small or large number of equities.
    Attributes
    ----------
    The table with which this loader interacts contains the following
    attributes:
    first_row : dict
        Map from asset_id -> index of first row in the dataset with that id.
    last_row : dict
        Map from asset_id -> index of last row in the dataset with that id.
    calendar_offset : dict
        Map from asset_id -> calendar index of first row.
    start_session_ns: int
        Epoch ns of the first session used in this dataset.
    end_session_ns: int
        Epoch ns of the last session used in this dataset.
    calendar_name: str
        String identifier of trading calendar used (ie, "NYSE").
    We use first_row and last_row together to quickly find ranges of rows to
    load when reading an asset's data into memory.
    We use calendar_offset and calendar to orient loaded blocks within a
    range of queried dates.
    Notes
    ------
    A Bcolz CTable is comprised of Columns and Attributes.
    The table with which this loader interacts contains the following columns:
    ['open', 'high', 'low', 'close', 'volume', 'day', 'id'].
    The data in these columns is interpreted as follows:
    - Price columns ('open', 'high', 'low', 'close') are interpreted as 1000 *
      as-traded dollar value.
    - Volume is interpreted as as-traded volume.
    - Day is interpreted as seconds since midnight UTC, Jan 1, 1970.
    - Id is the asset id of the row.
    The data in each column is grouped by asset and then sorted by day within
    each asset block.
    The table is built to represent a long time range of data, e.g. ten years
    of equity data, so the lengths of each asset block is not equal to each
    other. The blocks are clipped to the known start and end date of each asset
    to cut down on the number of empty values that would need to be included to
    make a regular/cubic dataset.
    When read across the open, high, low, close, and volume with the same
    index should represent the same asset and day.
    See Also
    --------
    zipline.data.us_equity_pricing.BcolzDailyBarWriter
    """

    _dtypes = {
        "adjusted_underlying_close": float64_dtype,
        "ask": float64_dtype,
        "bid": float64_dtype,
        "close": float64_dtype,
        "delta": float64_dtype,
        "gamma": float64_dtype,
        "interest_rate": float64_dtype,
        "implied_volatility": float64_dtype,
        "mid": float64_dtype,
        "moneyness": float64_dtype,
        "option_value": float64_dtype,
        "rho": float64_dtype,
        "spread": float64_dtype,
        "statistical_volatility": float64_dtype,
        "strike_price": float64_dtype,
        "theta": float64_dtype,
        "unadjusted_underlying_close": float64_dtype,
        "vega": float64_dtype,
        "open_interest": uint32_dtype,
        "volume": uint32_dtype,
        "days_to_expiration": uint32_dtype,
        "id": uint32_dtype,
        "day": uint32_dtype,
    }

    def __init__(self, table, read_all_threshold=3000):
        self._maybe_table_rootdir = table
        # Cache of fully read np.array for the carrays in the daily bar table.
        # raw_array does not use the same cache, but it could.
        # Need to test keeping the entire array in memory for the course of a
        # process first.
        self._spot_cols = {}
        self.PRICE_ADJUSTMENT_FACTOR = 0.001
        self._read_all_threshold = read_all_threshold

    @lazyval
    def _table(self):
        maybe_table_rootdir = self._maybe_table_rootdir
        if isinstance(maybe_table_rootdir, ctable):
            return maybe_table_rootdir
        return ctable(rootdir=maybe_table_rootdir, mode="r")

    @lazyval
    def sessions(self):
        if "calendar" in self._table.attrs.attrs:
            # backwards compatibility with old formats, will remove
            return DatetimeIndex(self._table.attrs["calendar"], tz="UTC")
        else:
            cal = get_calendar(self._table.attrs["calendar_name"])
            start_session_ns = self._table.attrs["start_session_ns"]
            start_session = Timestamp(start_session_ns, tz="UTC")

            end_session_ns = self._table.attrs["end_session_ns"]
            end_session = Timestamp(end_session_ns, tz="UTC")

            sessions = cal.sessions_in_range(start_session, end_session)

            return sessions

    @lazyval
    def _first_rows(self):
        return {
            int(asset_id): start_index
            for asset_id, start_index in iteritems(self._table.attrs["first_row"])
        }

    @lazyval
    def _last_rows(self):
        return {
            int(asset_id): end_index
            for asset_id, end_index in iteritems(self._table.attrs["last_row"])
        }

    @lazyval
    def _calendar_offsets(self):
        return {
            int(id_): offset
            for id_, offset in iteritems(self._table.attrs["calendar_offset"])
        }

    @lazyval
    def first_trading_day(self):
        try:
            return Timestamp(self._table.attrs["first_trading_day"], unit="s", tz="UTC")
        except KeyError:
            return None

    @lazyval
    def trading_calendar(self):
        if "calendar_name" in self._table.attrs.attrs:
            return get_calendar(self._table.attrs["calendar_name"])
        else:
            return None

    @property
    def last_available_dt(self):
        return self.sessions[-1]

    def _compute_slices(self, start_idx, end_idx, assets):
        """
        Compute the raw row indices to load for each asset on a query for the
        given dates after applying a shift.
        Parameters
        ----------
        start_idx : int
            Index of first date for which we want data.
        end_idx : int
            Index of last date for which we want data.
        assets : pandas.Int64Index
            Assets for which we want to compute row indices
        Returns
        -------
        A 3-tuple of (first_rows, last_rows, offsets):
        first_rows : np.array[intp]
            Array with length == len(assets) containing the index of the first
            row to load for each asset in `assets`.
        last_rows : np.array[intp]
            Array with length == len(assets) containing the index of the last
            row to load for each asset in `assets`.
        offset : np.array[intp]
            Array with length == (len(asset) containing the index in a buffer
            of length `dates` corresponding to the first row of each asset.
            The value of offset[i] will be 0 if asset[i] existed at the start
            of a query.  Otherwise, offset[i] will be equal to the number of
            entries in `dates` for which the asset did not yet exist.
        """
        # The core implementation of the logic here is implemented in Cython
        # for efficiency.
        return _compute_row_slices(
            self._first_rows,
            self._last_rows,
            self._calendar_offsets,
            start_idx,
            end_idx,
            assets,
        )

    def load_raw_arrays(self, columns, start_date, end_date, assets):
        # Assumes that the given dates are actually in calendar.
        start_idx = self.sessions.get_loc(start_date)
        end_idx = self.sessions.get_loc(end_date)
        first_rows, last_rows, offsets = self._compute_slices(
            start_idx, end_idx, assets
        )
        read_all = len(assets) > self._read_all_threshold
        return _read_bcolz_data(
            self._table,
            (end_idx - start_idx + 1, len(assets)),
            list(columns),
            first_rows,
            last_rows,
            offsets,
            read_all,
        )

    def _spot_col(self, colname):
        """
        Get the colname from daily_bar_table and read all of it into memory,
        caching the result.
        Parameters
        ----------
        colname : string
            A name of a OHLCV carray in the daily_bar_table
        Returns
        -------
        array (uint32)
            Full read array of the carray in the daily_bar_table with the
            given colname.
        """
        try:
            col = self._spot_cols[colname]
        except KeyError:
            col = self._spot_cols[colname] = self._table[colname]
        return col

    def get_last_traded_dt(self, asset, day):
        volumes = self._spot_col("volume")

        search_day = day

        while True:
            try:
                ix = self.sid_day_index(asset, search_day)
            except NoDataBeforeDate:
                return NaT
            except NoDataAfterDate:
                prev_day_ix = self.sessions.get_loc(search_day) - 1
                if prev_day_ix > -1:
                    search_day = self.sessions[prev_day_ix]
                continue
            except NoDataOnDate:
                return NaT
            if volumes[ix] != 0:
                return search_day
            prev_day_ix = self.sessions.get_loc(search_day) - 1
            if prev_day_ix > -1:
                search_day = self.sessions[prev_day_ix]
            else:
                return NaT

    def sid_day_index(self, sid, day):
        """
        Parameters
        ----------
        sid : int
            The asset identifier.
        day : datetime64-like
            Midnight of the day for which data is requested.
        Returns
        -------
        int
            Index into the data tape for the given sid and day.
            Raises a NoDataOnDate exception if the given day and sid is before
            or after the date range of the equity.
        """
        try:
            day_loc = self.sessions.get_loc(day)
        except:
            raise NoDataOnDate(
                "day={0} is outside of calendar={1}".format(day, self.sessions)
            )
        offset = day_loc - self._calendar_offsets[sid]
        if offset < 0:
            raise NoDataBeforeDate(
                "No data on or before day={0} for sid={1}".format(day, sid)
            )
        ix = self._first_rows[sid] + offset
        if ix > self._last_rows[sid]:
            raise NoDataAfterDate(
                "No data on or after day={0} for sid={1}".format(day, sid)
            )
        return ix

    def get_value(self, sid, dt, field):
        """
        Parameters
        ----------
        sid : int
            The asset identifier.
        day : datetime64-like
            Midnight of the day for which data is requested.
        colname : string
            The price field. e.g. ('open', 'high', 'low', 'close', 'volume')
        Returns
        -------
        float
            The spot price for colname of the given sid on the given day.
            Raises a NoDataOnDate exception if the given day and sid is before
            or after the date range of the equity.
            Returns -1 if the day is within the date range, but the price is
            0.
        """
        ix = self.sid_day_index(sid, dt)
        price = self._spot_col(field)[ix]
        if field not in ("volume", "open_interest", "days_to_expiration"):
            if price == 0:
                return nan
            else:
                return round(price * 0.001, 4)
        else:
            return price

    def get_chain(self, options, dt):
        """Given a root sybmol and a valid date, return an option chain

        Parameters
        ----------
        options : str
            Entire universe of options from which to select those active on `dt`
        dt : pd.Timestamp
            Valid session from which all options are retrieved from the tape

        Returns
        -------

        """
        columns = [
            col
            for col in OPTION_PRICING_BCOLZ_COLUMNS
            if col not in ("delta", "gamma", "rho", "theta", "vega")
        ] + ["delta", "gamma", "rho", "theta", "vega"]

        series = []
        raw_arrays = self.load_raw_arrays(columns, dt, dt, options)
        for i in range(len(columns)):
            col = columns[i]
            series.append(pd.Series(
                raw_arrays[i][0],
                name=col,
                dtype=BcolzDailyChainReader._dtypes[col]
            ))

        df = pd.concat(series, axis=1)

        # it's probably risk to use the mid price as the column to consider if
        # there's a chain or not. argument could be made if there's not even a
        # mid price there's no market and it's not worth having the chain in the
        # analysis
        mask = df.mid.notnull()
        option_chain = df[mask]

        option_chain.rename(columns={
            "id": "sid",
            "day": "date"
        }, inplace=True)

        option_chain.date = option_chain.date.apply(
            lambda x: pd.Timestamp(x, unit="s", tz="UTC")
        )

        return option_chain

    def currency_codes(self, sids):
        # XXX: This is pretty inefficient. This reader doesn't really support
        # country codes, so we always either return USD or None if we don't
        # know about the sid at all.
        first_rows = self._first_rows
        out = []
        for sid in sids:
            if sid in first_rows:
                out.append("USD")
            else:
                out.append(None)
        return np.array(out, dtype=object)
