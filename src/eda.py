import abc
import pathlib

import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as tsa

import src.visualization as vis

# TODO should rename this module to data

DATA_DIR = pathlib.Path(__file__).parent.resolve() / ".." / "data"


class TimeSeries(abc.ABC):
    def __init__(self, name: str, expected_seasonality: int, differencing_periods: int = 1):
        """
        :param name:
        :param expected_seasonality:
        :param differencing_periods:
        """
        self.name = name
        self.expected_seasonality = expected_seasonality
        self.differencing_periods = differencing_periods
        self.original: pd.Series | None = None
        self.diffed: pd.Series | None = None
        self.log: pd.Series | None = None
        self.log_diffed: pd.Series | None = None

    @abc.abstractmethod
    def load_data(self, fraction: float = 0.8, left_side: bool = True):
        """
        Load the dataset and populate the time series fields

        :param fraction: fraction of the time series to load
        :param left_side: True to load the left side of the data, i.e. from the beginning,
            False to take the portion from the end. This is useful to determine whether to load
            the train or test data
        """
        pass

    def plot_visualizations(
            self,
            original: bool = False,
            diffed: bool = False,
            log: bool = False,
            log_diffed: bool = False
    ):
        # TODO maybe add a "path" param to save the images if provided
        if original:
            self._plot_ts_visulizations(self.original, ts_name="Original")
        if diffed:
            self._plot_ts_visulizations(self.diffed, ts_name=f"{self.differencing_periods}-Differenced")
        if log:
            self._plot_ts_visulizations(self.log, ts_name="Log")
        if log_diffed:
            self._plot_ts_visulizations(self.log_diffed, ts_name=f"Log {self.differencing_periods}-Differenced")

    def _plot_ts_visulizations(self, ts: pd.Series, ts_name: str):
        vis.ts_eda(ts=ts, ts_name=f"{self.name} - {ts_name}", expected_seasonality=self.expected_seasonality)

    def stationarity_tests(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                # first two elements of adfuller and kpss are statistic, p-value
                ["original", "adfuller", *(tsa.adfuller(self.original)[:2])],
                ["diffed", "adfuller", *(tsa.adfuller(self.log)[:2])],
                ["log", "adfuller", *(tsa.adfuller(self.diffed)[:2])],
                ["log_diffed", "adfuller", *(tsa.adfuller(self.log_diffed)[:2])],

                ["original", "kpss", *(tsa.kpss(self.original)[:2])],
                ["diffed", "kpss", *(tsa.kpss(self.log)[:2])],
                ["log", "kpss", *(tsa.kpss(self.diffed)[:2])],
                ["log_diffed", "kpss", *(tsa.kpss(self.log_diffed)[:2])],
            ],
            columns=["ts", "test", "statistic", "p-value"]
        )


class SMH(TimeSeries):
    def __init__(self, differencing_periods: int = 1):
        # expected_seasonality=252 because number of trading days in normal year
        super().__init__(name="SMH", expected_seasonality=252, differencing_periods=differencing_periods)

    def load_data(self, fraction: float = 0.8, left_side: bool = True):
        smh = pd.read_csv(DATA_DIR / 'smh-2000-06-05_2024-08-23.csv', parse_dates=True, index_col=0)

        # basically a train-test split
        smh = _fraction_ts(smh, fraction, left_side)

        # Apparently it is better to use DatetimeIndex with irregularly sampled ts,
        #   which is my case since trading week is 5 days out of 7
        # smh.index = smh.index.to_period('D')

        self.original = smh["Close"]
        self._clean_original_ts()

        self.diffed = self.original.diff(periods=self.differencing_periods).dropna()
        self.log = self.original.apply(np.log)
        self.log_diffed = self.log.diff(periods=self.differencing_periods).dropna()

    def _clean_original_ts(self):
        self.original = self.original.drop(self.original.index[self.original.index.year == 2000])

    def years_cardinality(self):
        """
        Ideally useful info because trading weeks consist of 5 days, Monday to Friday, meaning
            that years aren't 365 days, more like ~250
        """

        yearly_cardinality = []
        for year in self.original.index.year.unique():
            yearly_cardinality.append({
                "year": year,
                "cardinality": np.sum(self.original.index.year == year)
            })

        cardinality_df = pd.DataFrame.from_records(yearly_cardinality)
        return cardinality_df


class SemiconductorSales(TimeSeries):
    def __init__(self, differencing_periods: int = 1):
        super().__init__(name="SemiconductorSales", expected_seasonality=12, differencing_periods=differencing_periods)

    def load_data(self, fraction: float = 0.8, left_side: bool = True):
        sales = pd.read_csv(DATA_DIR / 'global-semiconductor-sales-2012-2024.csv', parse_dates=True, index_col=0).dropna()

        # basically a train-test split
        sales = _fraction_ts(sales, fraction, left_side)

        sales.index = sales.index.to_period('M')

        self.original = sales["sales"]
        self.diffed = self.original.diff(periods=self.differencing_periods).dropna()
        self.log = self.original.apply(np.log)
        self.log_diffed = self.log.diff(periods=self.differencing_periods).dropna()


class SMHForSemiconductorSales(SMH):
    def __init__(self, differencing_periods: int = 1):
        super().__init__(differencing_periods=differencing_periods)
        self.name = "SMHForSemiconductorSales"
        self.expected_seasonality = 12  # resampled with monthly freq

    def load_data(self, fraction: float = 0.8, left_side: bool = True):
        # `fraction` will be applied after resampling so that the length of the
        #   resampled smh time series and the sales time series match
        super().load_data(fraction=1.0, left_side=left_side)

        # Resample each SMH time series and take only data points in the
        #   date range [2012-01-01, 2024-01-01], which is the date range of the sales TS
        # Also, the first `self.differencing_periods` are skipped in diffed series because differencing is applied
        #   on SMH before resampling, meaning that the size of the resampled ts
        #   would be `self.differencing_periods` longer otherwise
        sales_ts_start_date, sales_ts_end_date = '2012-01-01', '2024-01-02'
        self.original = self.original[sales_ts_start_date:sales_ts_end_date].resample("MS").first()
        self.original = _fraction_ts(self.original, fraction, left_side)
        self.original.index = self.original.index.to_period("M")

        self.diffed = self.diffed[sales_ts_start_date:sales_ts_end_date].resample("MS").first()[self.differencing_periods:]
        self.diffed = _fraction_ts(self.diffed, fraction, left_side)
        self.diffed.index = self.diffed.index.to_period("M")

        self.log = self.log[sales_ts_start_date:sales_ts_end_date].resample("MS").first()
        self.log = _fraction_ts(self.log, fraction, left_side)
        self.log.index = self.log.index.to_period("M")

        self.log_diffed = self.log_diffed[sales_ts_start_date:sales_ts_end_date].resample("MS").first()[self.differencing_periods:]
        self.log_diffed = _fraction_ts(self.log_diffed, fraction, left_side)
        self.log_diffed.index = self.log_diffed.index.to_period("M")



def _fraction_ts(ts: pd.Series | pd.DataFrame, fraction: float = 0.8, left_side: bool = True):
    if fraction == 1.0:
        return ts

    # basically a train-test split
    n_samples = int(ts.shape[0] * fraction)
    fractioned = ts[:n_samples] if left_side else ts[(ts.shape[0] - n_samples):]
    return fractioned




