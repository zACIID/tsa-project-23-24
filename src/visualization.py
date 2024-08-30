import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.plotting as pd_plt
import seaborn as sns
import statsmodels.graphics.tsaplots as tsa_plt
import statsmodels.tsa.seasonal as tsa_season
import statsmodels.tsa.stattools as tsa

DEFAULT_FIG_SIZE = (12, 8)


def plot_ts(ts: pd.Series, title: str = "Time series") -> plt.Figure:
    ax = ts.plot(figsize=DEFAULT_FIG_SIZE)
    ax.figure.suptitle(title)
    return ax.figure


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

    # Required else plot breaks
    if isinstance(ts.index, pd.PeriodIndex):
        ts = ts.copy()
        ts.index = ts.index.to_timestamp()

    # Create the plot
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.set_title('Time Series with Rolling Mean and Std')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    sns.lineplot(ax=ax, data=ts, label='Time Series')
    sns.lineplot(ax=ax, data=rolling_mean, label=f'Rolling Mean (window={k})')
    sns.lineplot(ax=ax, data=rolling_std, label=f'Rolling Std (window={k})')

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

    if isinstance(ts.index, pd.PeriodIndex):
        ts = ts.copy()
        ts.index = ts.index.to_timestamp()

    if stl:
        stl = tsa_season.STL(
            ts,
            period=period,
            seasonal=seasonality,
        )
        res = stl.fit()
        fig = res.plot()
        fig.suptitle(f"STL decomposition - Seasonality={seasonality}")
    else:
        fig = tsa_season.seasonal_decompose(
            ts, model='additive',
            period=seasonality
        ).plot()
        fig.suptitle(f"MA-based decomposition - Seasonality={seasonality}")

    fig.set_size_inches(DEFAULT_FIG_SIZE)
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
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    sns.boxplot(ax=ax, data=seasons_data)

    # Adding title and labels
    ax.set_title(f'Annual seasonality boxplots (monthly distributions)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Value')

    return fig


def plot_acf_pacf(ts: pd.Series) -> plt.Figure:
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(16, 24))

    # This first plot should be a little bit more zoomed out
    # I saw a heuristic on some StackExchange discussions that said to plot lags
    #   until time_series.length/100 or until the ACF goes to 0, whichever of the two comes first
    tsa_plt.plot_acf(ts, ax=axs[0], lags=ts.shape[0] // 100, auto_ylims=True)
    xlim_low, xlim_high = axs[0].get_xlim()
    axs[0].set_xticks(np.arange(0, xlim_high, 5))

    # These two are automatically zoomed-in by the plotting library, based on the behavior of ACF values
    tsa_plt.plot_acf(ts, ax=axs[1], auto_ylims=True)
    xlim_low, xlim_high = axs[1].get_xlim()
    axs[1].set_xticks(np.arange(0, xlim_high, 5))

    tsa_plt.plot_pacf(ts, ax=axs[2], auto_ylims=True)
    xlim_low, xlim_high = axs[2].get_xlim()
    axs[2].set_xticks(np.arange(0, xlim_high, 5))

    return fig


def plot_top_k_autocorr_lags(ts: pd.Series, k: int = 10) -> plt.Figure:
    # Extract the top k autocorr scores
    acf_res, _ = tsa.acf(ts, nlags=100, alpha=0.05)

    # From 1 to k + 1 because lag 0 is not interesting
    top_lags = np.argsort(-np.abs(acf_res))[1:k+1]
    top_autocorr = pd.DataFrame({
        "lag": top_lags,
        "autocorr": acf_res[top_lags]
    })

    # Plot every lag except 0 - also, the first chart of the grid is the autocorr barplot
    to_plot = top_autocorr["lag"]
    nrows, ncols = len(to_plot) // 2 + 1, 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*10, nrows*5))
    fig.suptitle(f"Top {k} autocorrelation lags")

    # Barplot of the top autocorrelations
    sns.barplot(ax=axs[0, 0], data=top_autocorr, x="lag", y="autocorr", color="skyblue")

    for i, lag in enumerate(to_plot):
        i += 1  # because first axes is occupied by the top-autocorr chart
        pd_plt.lag_plot(ts, ax=axs[i // ncols, i % ncols], lag=lag)

    return fig
