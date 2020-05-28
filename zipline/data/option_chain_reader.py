import numpy as np
import pandas as pd
from zipline.data.session_bars import SessionBarReader


class OptionChainSessionChainReader(SessionBarReader):

    def __init__(self, chain_reader):
        self._chain_reader = chain_reader

    def load_raw_arrays(self, columns, start_date, end_date, assets):
        """
        Parameters
        ----------
        fields : list of str
            'sid'
        start_dt: Timestamp
           Beginning of the window range.
        end_dt: Timestamp
           End of the window range.
        sids : list of int
           The asset identifiers in the window.

        Returns
        -------
        list of np.ndarray
            A list with an entry per field of ndarrays with shape
            (minutes in range, sids) with a dtype of float64, containing the
            values for the respective field over start and end dt range.
        """
        num_sessions = len(
            self.trading_calendar.sessions_in_range(start_date, end_date)
        )
        shape = num_sessions, len(assets)

        results = []

        tc = self._chain_reader.trading_calendar
        sessions = tc.sessions_in_range(start_date, end_date)

        # Get partitions
        partitions_by_asset = {}
        for asset in assets:
            partitions = []
            partitions_by_asset[asset] = partitions

            start = start_date
            start_loc = sessions.get_loc(start)

            end = end_date
            end_loc = len(sessions) - 1

            partitions.append((asset.sid, start, end, start_loc, end_loc))

        for column in columns:
            if column != 'volume' and column != 'sid':
                out = np.full(shape, np.nan)
            else:
                out = np.zeros(shape, dtype=np.int64)

            for i, asset in enumerate(assets):
                partitions = partitions_by_asset[asset]

                for sid, start, end, start_loc, end_loc in partitions:
                    if column != 'sid':
                        result = self._chain_reader.load_raw_arrays(
                            [column], start, end, [sid])[0][:, 0]
                    else:
                        result = int(sid)
                    out[start_loc:end_loc + 1, i] = result

            results.append(out)

        return results

    @property
    def last_available_dt(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The last session for which the reader can provide data.
        """
        return self._chain_reader.last_available_dt

    @property
    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """
        return self._chain_reader.trading_calendar

    @property
    def first_trading_day(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The first trading day (session) for which the reader can provide
            data.
        """
        return self._chain_reader.first_trading_day

    def get_value(self, asset, dt, field):
        """
        Retrieve the value at the given coordinates.

        Parameters
        ----------
        asset : zipline.asset.Asset
            The asset identifier.
        dt : pd.Timestamp
            The timestamp for the desired data point.
        field : string
            The OHLVC name for the desired data point.

        Returns
        -------
        value : float|int
            The value at the given coordinates, ``float`` for OHLC, ``int``
            for 'volume'.

        Raises
        ------
        NoDataOnDate
            If the given dt is not a valid market minute (in minute mode) or
            session (in daily mode) according to this reader's tradingcalendar.
        """
        return self._chain_reader.get_value(asset, dt, field)

    def get_last_traded_dt(self, asset, dt):
        """
        Get the latest minute on or before ``dt`` in which ``asset`` traded.

        If there are no trades on or before ``dt``, returns ``pd.NaT``.

        Parameters
        ----------
        asset : zipline.asset.Asset
            The asset for which to get the last traded minute.
        dt : pd.Timestamp
            The minute at which to start searching for the last traded minute.

        Returns
        -------
        last_traded : pd.Timestamp
            The dt of the last trade for the given asset, using the input
            dt as a vantage point.
        """
        pass
        # if asset is None:
        #     return pd.NaT
        # contract = rf.asset_finder.retrieve_asset(sid)
        # return self._chain_reader.get_last_traded_dt(contract, dt)

    @property
    def sessions(self):
        """
        Returns
        -------
        sessions : DatetimeIndex
           All session labels (unioning the range for all assets) which the
           reader can provide.
        """
        return self._chain_reader.sessions
