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


def plot_ts(ts: pd.Series, title: str = "Time series") -> plt.Figure:
    fig = ts.plot(figsize=(16, 12))
    fig.set_title(title)
    return fig


def plot_time_series_with_rolling_stats(ts: pd.Series, k: int) -> plt.Figure:
    """
    Plots the original time series along with its rolling mean and rolling std.

    :param ts: The time series data with a PeriodIndex.
    :param k: The window size for computing the rolling statistics.
    :return: figure
    """

    # Compute rolling statistics
    rolling_mean = ts.rolling(window=k).mean()
    rolling_std = ts.rolling(window=k).std()

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_title('Time Series with Rolling Mean and Std')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    sns.lineplot(ax=ax, data=ts, label='Time Series')
    sns.lineplot(ax=ax, data=rolling_mean, label=f'Rolling Mean (window={k})')
    sns.lineplot(ax=ax, data=rolling_std, label=f'Rolling Std (window={k})')

    # Show the legend
    fig.legend()

    return fig


def plot_ts_decomposition(
        ts: pd.Series,
        period: int,
        seasonality: int,
        stl: bool = True,
) -> plt.Figure:
    """
    Performs either STL or MA-based (additive) decomposition, based on the specified flag.

    :param ts: time series
    :param period: number
    :param seasonality: time series
    :param stl: whether to go with STL or MA-based decomposition
    :return:
    """
    if stl:
        stl = tsa_season.STL(
            ts,
            period=period,
            seasonality=seasonality,
        )
        res = stl.fit()
        fig = res.plot()
    else:
        fig = tsa_season.seasonal_decompose(
            ts, model='additive',
            period=seasonality
        ).plot()

    fig.set_size_inches(16, 12)
    return fig

def plot_annual_season_boxplots(ts: pd.Series) -> plt.Figure:
    seasons_data = {}

    # This will be useful only if I decide to plot seasons of arbitrary number of days
    # for i in range(1, seasonality+1):
    #     season_label = f'Period {i+1}'
    #     # season_data = ts.resample(f"{i}B", offset=f"{i}B")
    #     season_data = ts.resample(f"{seasonality}M", offset=pd.Timedelta(i)).first()
    #     seasons_data[season_label] = season_data.values

    for month in range(1, 13):
        month_label = f'Month {month}'
        seasons_data[month_label] = ts[ts.index.month == month].values

    # Plot the boxplot for each season
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.boxplot(ax=ax, data=seasons_data)

    # Adding title and labels
    ax.set_title(f'Annual seasonality boxplots (monthly distributions)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Value')

    return fig


def plot_acf_pacf(ts: pd.Series) -> plt.Figure:
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 24))
    tsa_plt.plot_acf(ts, ax=axs[0], lags=ts.shape[0] // 100, auto_ylims=True)
    tsa_plt.plot_acf(ts, ax=axs[0], auto_ylims=True)
    tsa_plt.plot_pacf(ts, ax=axs[0], auto_ylims=True)

    return fig


def plot_top_k_autocorr_lags(ts: pd.Series, k: int = 10) -> plt.Figure:
    # Extract the top k autocorr scores
    acf_res, _ = tsa.acf(ts, nlags=40, alpha=0.05)

    top_autocorr = pd.DataFrame({
        # k + 1 because lag 0 is not really interesting
        "lag": np.argsort(-np.abs(acf_res))[:k+1],
        "autocorr": np.sort(-np.abs(acf_res))[:k+1]
    })

    # print(np.argsort(-np.abs(acf_res))[1:k+1])
    # print(np.sort(-np.abs(acf_res))[1:11])

    # Plot every lag except 0
    to_plot = top_autocorr["lag"]
    nrows, ncols = len(to_plot) // 2 + 1, 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*10, nrows*5))
    fig.suptitle(f"Top {k} autocorrelation lags")

    for i, lag in enumerate(to_plot[1:]):
        pd_plt.lag_plot(ts, ax=axs[i // ncols, i % ncols], lag=lag)

    return fig
