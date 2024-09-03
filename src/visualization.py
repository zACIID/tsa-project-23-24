import typing

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


def plot_time_series_with_rolling_stats(ts: pd.Series, k: int, ts_name: str = "") -> plt.Figure:
    """
    Plots the original time series along with its rolling mean and rolling std.

    :param ts: The time series data with a PeriodIndex.
    :param k: The window size for computing the rolling statistics.
    :param ts_name: label for the time series
    :return: figure
    """

    # Required else plot breaks
    if isinstance(ts.index, pd.PeriodIndex):
        ts = ts.copy()
        ts.index = ts.index.to_timestamp()

    # Compute rolling statistics
    rolling_mean = ts.rolling(window=k).mean().dropna()
    rolling_std = ts.rolling(window=k).std().dropna()

    # Create the plot
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    name = ts_name if len(ts_name) > 0 else 'Time Series'
    ax.set_title(f"{name} with Rolling Mean and Std")
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    sns.lineplot(ax=ax, data=ts, label=ts_name)
    sns.lineplot(ax=ax, data=rolling_mean, label=f'Rolling Mean (window={k})')
    sns.lineplot(ax=ax, data=rolling_std, label=f'Rolling Std (window={k})')

    return fig


def plot_ts_decomposition(
        ts: pd.Series,
        expected_seasonality: int,
        seasonal_smoother_length: int = 7,
        stl: bool = True,
) -> plt.Figure:
    """
    Performs either STL or MA-based (additive) decomposition, based on the specified flag.

    :param ts: time series
    :param expected_seasonality: period/seasonality of the ts
    :param seasonal_smoother_length:
    :param stl: whether to go with STL or MA-based decomposition
    :return:
    """

    if isinstance(ts.index, pd.PeriodIndex):
        ts = ts.copy()
        ts.index = ts.index.to_timestamp()

    if stl:
        stl = tsa_season.STL(
            ts,
            period=expected_seasonality,
            seasonal=7,
        )
        res = stl.fit()
        fig = res.plot()
        fig.suptitle(f"STL decomposition - Seasonality={expected_seasonality}")
    else:
        fig = tsa_season.seasonal_decompose(
            ts, model='additive',
            period=seasonal_smoother_length
        ).plot()
        fig.suptitle(f"MA-based decomposition - Seasonality={expected_seasonality}")

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


def plot_acf_pacf(ts: pd.Series, expected_seasonality: int) -> plt.Figure:
    """
    :param ts:
    :param expected_seasonality: this determines the number of lags shown in the first ACF plot,
        which is a zoomed-out version to help detect seasonality or other long-term patterns
    :return:
    """
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(16, 24))

    expected_seasonality = 10 if expected_seasonality == 1 else expected_seasonality

    zoomed_out_lags = expected_seasonality*3 if ts.shape[0] > expected_seasonality*3 else min(ts.shape[0], expected_seasonality*2)
    tsa_plt.plot_acf(
        ts,
        ax=axs[0],
        lags=zoomed_out_lags,
        auto_ylims=True
    )
    xlim_low, xlim_high = axs[0].get_xlim()
    axs[0].set_xticks(np.arange(0, xlim_high, expected_seasonality//2))

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


def ts_eda(
        ts: pd.Series,
        ts_name: str,
        expected_seasonality: int,
        ts_plot: bool = True,
        rolling_stats_plot: bool = True,
        season_boxplots: bool = True,
        ts_decomposition: bool = True,
        acf_pacf_plots: bool = True,
        top_k_autocorr_plots: bool = True,
):
    """
    Collection of plots for the provided time series. Can enable/disable any of them by providing the appropriate boolean
    :param ts:
    :param ts_name:
    :param expected_seasonality:
    :param ts_plot:
    :param rolling_stats_plot:
    :param season_boxplots:
    :param ts_decomposition:
    :param acf_pacf_plots:
    :param top_k_autocorr_plots:
    """

    # NOTE: fig.show()  # figure is non-interactive and thus cannot be shown
    # assigning plots to fig because otherwise plot may be shown two times

    if ts_plot:
        _ = plot_ts(ts, title=ts_name)

    if rolling_stats_plot:
        # Absolutely arbitrary rule to decide rolling window period§
        _ = plot_time_series_with_rolling_stats(ts=ts, k=max(10, min(int(ts.shape[0]*0.01), 50)), ts_name=ts_name)

    if season_boxplots:
        _ = plot_annual_season_boxplots(ts)

    if ts_decomposition:
        _ = plot_ts_decomposition(
            ts,
            expected_seasonality=expected_seasonality,
            # seasonality must be odd because of STL (loess smoothing must be centered)
            stl=True
        )
        _ = plot_ts_decomposition(
            ts,
            expected_seasonality=expected_seasonality,
            stl=False
        )

    if acf_pacf_plots:
        _ = plot_acf_pacf(ts, expected_seasonality=expected_seasonality)

    if top_k_autocorr_plots:
        _ = plot_top_k_autocorr_lags(ts, k=10)


def plot_accf_grid(
        x: pd.Series,
        y: pd.Series,
        expected_seasonality: int = 12,
        x_name="x",
        y_name="y"
) -> plt.Figure:
    fig: plt.Figure = tsa_plt.plot_accf_grid(
        np.array([x, y]).T,
        negative_lags=True,
        lags=np.arange(-expected_seasonality*2, expected_seasonality*2 + 1, 1),
        adjusted=False
    )
    fig.set_size_inches(16, 12)
    axs = fig.axes
    axs[0].set_title(f"ACF - {x_name}")  # top-left
    axs[1].set_title(f"CCF - {x_name}_{{t+k}} & {y_name}_{{t}}")  # top-right
    axs[2].set_title(f"CCF - {y_name}_{{t+k}} & {x_name}_{{t}}")  # bot-left
    axs[3].set_title(f"ACF - {y_name}")  # bot-right

    fig.tight_layout()
    return fig


def plot_top_k_ccf_lags(x: pd.Series, y: pd.Series, k: int = 10, max_lag: int = 100) -> plt.Figure:
    """
    Plot the top-k cross-correlation lags between two time series x and y.

    Parameters:
    :param x: Predictor time series (to be lagged).
    :param y: Target time series.
    :param k: Number of top lags to consider.
    :param max_lag: Maximum lag to consider in CCF.
    :return: figure
    """
    # Compute the cross-correlation function (CCF) up to max_lag
    # y comes before x because we want negative lags, so we actually lag y with positive lags
    #   (tsa.ccf functions applies lags to the x var)
    ccf_res = tsa.ccf(x=y, y=x, adjusted=False)[:max_lag+1]

    # Extract the top k lags based on absolute cross-correlation values
    top_lags = np.argsort(-np.abs(ccf_res))[:k]
    top_ccf = pd.DataFrame({
        "lag": top_lags,
        "cross_corr": ccf_res[top_lags]
    })

    # Plot configuration
    nrows, ncols = len(top_lags) // 2 + 1, 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*10, nrows*5))
    fig.suptitle(f"Top {k} Cross-Correlation Lags between x and y")

    # Barplot of the top cross-correlations
    sns.barplot(ax=axs[0, 0], data=top_ccf, x="lag", y="cross_corr", color="skyblue")
    axs[0, 0].set_title("Top Cross-Correlations")

    # Generate cross-lag plots for the top lags
    for i, lag in enumerate(top_lags):
        i += 1  # because the first axes is occupied by the top-ccf barplot
        lagged_x = x.shift(-lag)  # lag X accordingly
        combined_df = pd.concat([lagged_x, y], axis=1).dropna()
        axs[i // ncols, i % ncols].scatter(combined_df.iloc[:, 0], combined_df.iloc[:, 1], alpha=0.7)

        # Lags are always < 0 due to how we calculated ccf
        axs[i // ncols, i % ncols].set_xlabel(f"x(t-{lag})")
        axs[i // ncols, i % ncols].set_ylabel("y(t)")
        axs[i // ncols, i % ncols].set_title(f"Lag {-lag}: CCF = {ccf_res[lag]:.2f}")

    fig.tight_layout()
    return fig


def cross_lag_plots(x: pd.Series, y: pd.Series, lags: np.ndarray):
    """
    Generate bivariate (cross-lag) plots of x_{t+h} against y_t for each lag in lags.

    Parameters:
    :param x: Predictor time series (to be lagged).
    :param y: Target time series.
    :param lags: List of lags to plot (positive or negative).
    """
    nrows, ncols = lags.shape[0] // 2 + 1, 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*10, nrows*5))

    if lags.shape[0] == 1:
        axs = [axs]

    for i, h in enumerate(lags):
        if h > 0:
            x_shifted = x.shift(-h)  # x_{t+h} means we need to shift X back by h
            label = f"x(t+{h})"
        elif h < 0:
            x_shifted = x.shift(-h)  # x_{t+h} with h negative means we need to shift x forward by |h|
            label = f"x(t{h})"
        else:
            x_shifted = x
            label = "x(t)"

        # Drop missing values due to the shift
        combined_df = pd.concat([x_shifted, y], axis=1).dropna()

        # Plot x_{t+h} against y_{t}
        ax = axs[i // ncols, i % ncols]
        ax.scatter(combined_df.iloc[:, 0], combined_df.iloc[:, 1], alpha=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("y(t)")
        ax.set_title(f'Cross Lag Plot: {label} vs y(t)')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
