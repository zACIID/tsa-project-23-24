import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as sk_metrics
import statsmodels.api as sm
import statsmodels.tsa.exponential_smoothing.ets as ets
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.statespace.mlemodel import PredictionResultsWrapper

import src.visualization as vis


def plot_predictions_from_fit_results(
        fit_results: ets.ETSResults | SARIMAXResults,
        train: pd.Series,
        test: pd.Series,
        alpha: float = 0.05,
        start_at: int = 0,
        zoom: float = 1.0
) -> plt.Figure:
    """
    :param fit_results:
    :param train:
    :param test:
    :param alpha:
    :param start_at: number of samples to start predicting from.
        Useful when there is seasonality and/or differencing involved,
        because the first seasonality+differencing periods can't be used for predictions
    :param zoom: see `plot_predictions` method
    :return:
    """
    # If 1-diffed, need to start from time 1, can't make predictions for time 0
    train = train[start_at:]

    # Difference between forecast and prediction methods:
    # https://stats.stackexchange.com/a/534159/395870
    # Basically get_prediction is the most general and complete one
    pred_res: ets.PredictionResults | PredictionResultsWrapper = fit_results.get_prediction(
        start=start_at,
        end=train.shape[0] + test.shape[0]
    )

    # Extract actual predictions
    # They are automatically calculated as in_sample for time points within the training data
    #   and as forecasts for the rest
    preds = pred_res.predicted_mean

    if isinstance(pred_res, ets.PredictionResultsWrapper):
        conf_int = pred_res.pred_int(alpha=alpha)
        conf_int_lower_name = "lower PI (alpha=%.6f)" % alpha
        conf_int_upper_name = "upper PI (alpha=%.6f)" % alpha
    elif isinstance(pred_res, PredictionResultsWrapper):
        conf_int = pred_res.conf_int(alpha=alpha)
        conf_int_lower_name = "lower y"
        conf_int_upper_name = "upper y"
    else:
        raise NotImplementedError("Unhandled type")

    in_sample, forecast = preds[:train.shape[0]], preds[-test.shape[0]:]
    in_sample_confint, forecast_confint = conf_int[:train.shape[0]], conf_int[-test.shape[0]:]

    return plot_predictions(
        train=train,
        test=test,
        forecast=forecast,
        forecast_confint=(forecast_confint[conf_int_lower_name], forecast_confint[conf_int_upper_name]),
        in_sample_preds=in_sample,
        in_sample_confint=(in_sample_confint[conf_int_lower_name], in_sample_confint[conf_int_upper_name]),
        zoom=zoom
    )


def plot_predictions(
        train: pd.Series,
        test: pd.Series,
        forecast: pd.Series,
        forecast_confint: typing.Tuple[pd.Series, pd.Series] = None,
        in_sample_preds: pd.Series = None,
        in_sample_confint: typing.Tuple[pd.Series, pd.Series] = None,
        zoom: float = 1.0,
) -> plt.Figure:
    """
    Plots the train time series, forecast time series, and optionally in-sample predictions,
    along with confidence intervals for the forecast and in-sample predictions.

    :param train: the actual training data
    :param test: the actual test data
    :param forecast: the forecasted values starting after the training data.
    :param forecast_confint: optional, the (lower, upper) confidence intervals for the forecast.  # TODO optional because cv_predict doesn't return confint
    :param in_sample_preds: optional, the (lower, upper) in-sample predictions matching the training data.
    :param in_sample_confint: optional, the confidence intervals for the in-sample predictions.
    :return: figure
    """

    if not (0 < zoom <= 1.0):
        raise ValueError("Zoom must be between 0 and 1.")

    if isinstance(train.index, pd.PeriodIndex):
        train = train.copy()
        train.index = train.index.to_timestamp()

    if isinstance(forecast.index, pd.PeriodIndex):
        forecast = forecast.copy()
        forecast.index = forecast.index.to_timestamp()

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=vis.DEFAULT_FIG_SIZE)

    # Plot the train time series
    sns.lineplot(ax=ax, x=train.index, y=train.values, label='Train', color='skyblue')

    # Plot the forecast time series and the actual
    sns.lineplot(ax=ax, x=test.index, y=test.values, label='Forecast Actual', color='indianred')
    sns.lineplot(ax=ax, x=test.index, y=forecast.values, label='Forecast', color='orange')

    # Plot the forecast confidence intervals
    if forecast_confint is not None:
        ax.fill_between(test.index,
                        forecast.values - forecast_confint[0],
                        forecast.values + forecast_confint[1],
                        color='orange', alpha=0.3, label='Forecast Conf. Int.')

    # Plot the in-sample predictions if provided
    if in_sample_preds is not None:
        sns.lineplot(ax=ax, x=train.index, y=in_sample_preds.values, label='In-sample Predictions',
                     color='green')

    # Plot the in-sample confidence intervals if provided
    if in_sample_confint is not None:
        ax.fill_between(train.index,
                        in_sample_preds.values - in_sample_confint[0],
                        in_sample_preds.values + in_sample_confint[1],
                        color='green', alpha=0.3, label='In-sample Conf. Int.')

    # Customize the x-axis  -> xlabels can be denser if zoom-in
    if 0 < zoom < 1.0:
        ax.set_xticks([
            pd.Timestamp(f'{year}-{month}-01')
            for year in np.concatenate([train.index.year.unique(), test.index.year.unique()])
            for month in np.arange(1, 13)
        ])
        ax.set_xticklabels([
            f"{year}-{month}"
            for year in np.concatenate([train.index.year.unique(), test.index.year.unique()])
            for month in np.arange(1, 13)
        ])
        ax.tick_params(rotation=90)
    # elif 0 < zoom < 0.2:
    #     # Customize the x-axis to show one tick per month, rotated 90degrees -> xlabels can be denser since zoom-in
    #     ax.set_xticks([
    #         pd.Timestamp(f'{year}-{month}-{day}')
    #         for year in np.concatenate([train.index.year.unique(), test.index.year.unique()])
    #         for month in np.arange(1, 13)
    #         for day in train.index[(train.index.year == year) & (train.index.month == month)].day.unique()
    #     ])
    #     ax.set_xticklabels([
    #         f"{year}-{month}-{day}"
    #         for year in np.concatenate([train.index.year.unique(), test.index.year.unique()])
    #         for month in np.arange(1, 13)
    #         for day in train.index[(train.index.year == year) & (train.index.month == month)].day.unique()
    #     ])
    #     ax.tick_params(rotation=90)
    else:
        # Customize the x-axis to show one tick per year, rotated 90degrees
        ax.set_xticks([pd.Timestamp(f'{year}-01-01') for year in np.concatenate([train.index.year.unique(), test.index.year.unique()])])
        ax.set_xticklabels([year for year in np.concatenate([train.index.year.unique(), test.index.year.unique()])])
        ax.tick_params(rotation=90)

    # Zoom in on the last part of the plot
    # Do this AFTER having set the xticks/labels
    if zoom < 1.0:
        xlim = ax.get_xlim()
        zoom_start = xlim[0] + (1 - zoom) * (xlim[1] - xlim[0])
        ax.set_xlim(left=zoom_start, right=test.index.values.max())

        # Adjust y-lim
        ts = np.concatenate([train, test])
        zoom_samples = int(ts.shape[0] * zoom)
        ylim_low, ylim_high = ts[-zoom_samples:].min(), ts[-zoom_samples:].max()
        ax.set_ylim(bottom=ylim_low-(ylim_low*0.05), top=ylim_high+(ylim_high*0.05))

    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Predictions')

    return fig


def get_diagnostics_df(
        fit_results: ets.ETSResults | SARIMAXResults
) -> pd.DataFrame:
    df = pd.DataFrame.from_records([{
        # Is AIC ever better than AICc?
        # https://stats.stackexchange.com/questions/319769/is-aicc-ever-worse-than-aic
        "AICc": fit_results.aicc,
        "BIC": fit_results.bic,
        "params_count": len(fit_results.params)
    }])
    return df


def get_forecast_error_df(test: pd.Series, forecast: pd.Series) -> pd.DataFrame:
    return pd.DataFrame.from_records([{
        "MAE": sk_metrics.mean_absolute_error(test, forecast),
        "RMSE": sk_metrics.root_mean_squared_error(test, forecast),
        "MAPE": sk_metrics.mean_absolute_percentage_error(test, forecast)
    }])


def get_stationarity_df(ts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            # first two elements of adfuller and kpss are statistic, p-value
            ["adfuller", *(sm.tsa.adfuller(ts)[:2])],
            ["kpss", *(sm.tsa.kpss(ts)[:2])],
        ],
        columns=["test", "statistic", "p-value"]
    )


def residuals_diagnostics(fitted: np.ndarray, residuals: np.ndarray) -> plt.Figure:
    # Plotting the diagnostics
    fig, ax = plt.subplots(2, 2, figsize=(20, 16))

    # 1. Residuals vs Fitted
    sns.scatterplot(x=fitted, y=residuals, ax=ax[0, 0])
    ax[0, 0].axhline(0, linestyle='--', color='r')
    ax[0, 0].set_title('Residuals vs Fitted')
    ax[0, 0].set_xlabel('Fitted values')
    ax[0, 0].set_ylabel('Residuals')

    # 2. Histogram (or KDE) of Residuals
    sns.histplot(residuals, kde=True, ax=ax[0, 1])
    ax[0, 1].set_title('Histogram of Residuals')

    # 3. Q-Q plot
    sm.qqplot(residuals, line='s', ax=ax[1, 0])
    ax[1, 0].set_title('Q-Q Plot')

    # 4. ACF plot of residuals
    sm.graphics.tsa.plot_acf(residuals, lags=30, ax=ax[1, 1])
    ax[1, 1].set_title('ACF of Residuals')

    # Show the plots
    fig.tight_layout()
    return fig
