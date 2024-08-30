import abc
import pathlib

import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as tsa

import src.visualization as vis

DATA_DIR = pathlib.Path(__file__).parent.resolve() / ".." / "data"


class TimeSeries(abc.ABC):
    def __init__(self, name: str, differencing_periods: int = 1):
        self.name = name
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
        # NOTE: fig.show()  # figure is non-interactive and thus cannot be shown
        # assigning plots to fig because otherwise plot may be shown two times

        fig = vis.plot_ts(ts, title=f"{self.name} - {ts_name}")
        fig = vis.plot_time_series_with_rolling_stats(ts, k=50)
        fig = vis.plot_annual_season_boxplots(ts)

        # seasonality must be odd because of STL (loess smoothing must be centered)
        # TODO parameterize this: expected_period, expected_seasonality as constructor args?
        fig = vis.plot_ts_decomposition(ts, period=252, seasonality=251, stl=True)
        fig = vis.plot_ts_decomposition(ts, period=252, seasonality=251, stl=False)

        fig = vis.plot_acf_pacf(ts)
        fig = vis.plot_top_k_autocorr_lags(ts, k=5)

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
        super().__init__(name="SMH", differencing_periods=differencing_periods)

    def load_data(self, fraction: float = 0.8, left_side: bool = True):
        smh = pd.read_csv(DATA_DIR / 'smh-2000-06-05_2024-08-23.csv', parse_dates=True, index_col=0)

        # basically a train-test split
        n_samples = int(smh.shape[0] * fraction)
        smh = smh[:n_samples] if left_side else smh[(smh.shape[0] - n_samples):]

        smh.index = smh.index.to_period('D')

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
        super().__init__(name="SemiconductorSales", differencing_periods=differencing_periods)

    def load_data(self, fraction: float = 0.8, left_side: bool = True):
        sales = pd.read_csv(DATA_DIR / 'global-semiconductor-sales-2012-2024.csv', parse_dates=True, index_col=0)

        # basically a train-test split
        n_samples = int(sales.shape[0] * fraction)
        sales = sales[:n_samples] if left_side else sales[(sales.shape[0] - n_samples):]

        sales.index = sales.index.to_period('M')

        self.original = sales["sales"]
        self.diffed = self.original.diff(periods=self.differencing_periods).dropna()
        self.log = self.original.apply(np.log)
        self.log_diffed = self.log.diff(periods=self.differencing_periods).dropna()

