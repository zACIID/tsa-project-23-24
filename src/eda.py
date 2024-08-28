import abc

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pandas.plotting as pd_plt
import pmdarima as pm
import pmdarima.model_selection as pm_modsel
import statsmodels.tsa.stattools as tsa
import statsmodels.graphics.tsaplots as tsa_plt
import statsmodels.tsa.arima_model as arima
import statsmodels.tsa.seasonal as tsa_season
import sklearn.model_selection as sk_modsel
import sklearn.metrics as sk_metrics
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.forecasting.stl import STLForecast, STLForecastResults

import visualization as vis

# TODO
"""
- an abstract class that loads a time series and exposes the original, diffed, log, logdiffed dataframes as properties
    - the differencing specified via __init__ param
    - has an abstract method "load_data" that loads the time series -> here I hardcode datasets in the child class
    - has a "plot_visualization" method that calls all the stuff needed from visualization.py
"""


class TimeSeries(abc.ABC):
    def __init__(self, name: str, differencing_periods: int = 1):
        self.name = name
        self.differencing_periods = differencing_periods
        self.original: pd.Series | None = None
        self.diffed: pd.Series | None = None
        self.log: pd.Series | None = None
        self.log_diffed: pd.Series | None = None

    @abc.abstractmethod
    def load_data(self):
        """
        Load the dataset and populate the time series fields
        """
        pass

    def plot_visualizations(self):
        # TODO maybe add a "path" param to save the images if provided
        self._plot_ts_visulizations(self.original, ts_name="Original")
        self._plot_ts_visulizations(self.diffed, ts_name=f"{self.differencing_periods}-Differenced")
        self._plot_ts_visulizations(self.original, ts_name="Log")
        self._plot_ts_visulizations(self.original, ts_name=f"Log {self.differencing_periods}-Differenced")

    def _plot_ts_visulizations(self, ts: pd.Series, ts_name: str):
        fig = vis.plot_ts(ts, title=f"{self.name} - {ts_name}")
        fig.show()
        fig = vis.plot_time_series_with_rolling_stats(ts, k=50)
        fig.show()
        fig = vis.plot_annual_season_boxplots(ts)
        fig.show()
        fig = vis.plot_acf_pacf(ts)
        fig.show()
        fig = vis.plot_top_k_autocorr_lags(ts, k=5)
        fig.show()

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
    def __init__(self):
        super().__init__(name="SMH")

    def load_data(self):
        smh = pd.read_csv('../data/smh-2000-06-05_2024-08-23.csv', parse_dates=True, index_col=0)
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

        yearly_cardinality = self.original.groupby(self.original.index.year).nunique()

        cardinality_df = yearly_cardinality.reset_index()
        cardinality_df.columns = ['year', 'cardinality']
        return cardinality_df


class SemiconductorSales(TimeSeries):
    def __init__(self):
        super().__init__(name="SemiconductorSales")

    def load_data(self):
        sales = pd.read_csv('../data/global-semiconductor-sales-2012-2024.csv', parse_dates=True, index_col=0)
        sales.index = sales.index.to_period('M')

        self.original = sales["sales"]
        self.diffed = self.original.diff(periods=self.differencing_periods).dropna()
        self.log = self.original.apply(np.log)
        self.log_diffed = self.log.diff(periods=self.differencing_periods).dropna()

