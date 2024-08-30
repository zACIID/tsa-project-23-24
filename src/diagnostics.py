import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as sk_metrics
import statsmodels.api as sm
import statsmodels.tsa.exponential_smoothing.ets as ets
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.statespace.mlemodel import PredictionResults


def plot_predictions_from_fit_results(
        fit_results: ets.ETSResults | ARIMAResults,
        train: pd.Series,
        test: pd.Series,
        alpha: float = 0.05
) -> plt.Figure:
    pred_res: ets.PredictionResults | PredictionResults = fit_results.get_prediction(
        start=0,
        end=train.shape[0] + test.shape[0]
    )

    # Extract actual predictions
    # They are automatically calculated as in_sample for time points within the training data
    #   and as forecasts for the rest
    preds = pred_res.predicted_mean

    if isinstance(pred_res, ets.PredictionResults):
        conf_int = pred_res.pred_int(alpha=alpha)
    elif isinstance(pred_res, PredictionResults):
        conf_int = pred_res.conf_int(alpha=alpha)
    else:
        raise NotImplementedError("Unhandled type")

    in_sample, forecast = preds[:train.shape[0]], preds[-test.shape[0]:]
    in_sample_confint, forecast_confint = conf_int[:train.shape[0]], conf_int[-test.shape[0]:]

    return plot_predictions(
        train=train,
        test=test,
        forecast=forecast,
        forecast_confint=forecast_confint,
        in_sample_preds=in_sample,
        in_sample_confint=in_sample_confint
    )


def plot_predictions(
        train: pd.Series,
        test: pd.Series,
        forecast: pd.Series,
        forecast_confint: pd.Series,
        in_sample_preds: pd.Series = None,
        in_sample_confint: pd.Series = None
) -> plt.Figure:
    """
    Plots the train time series, forecast time series, and optionally in-sample predictions,
    along with confidence intervals for the forecast and in-sample predictions.

    :param train: the actual training data
    :param test: the actual test data
    :param forecast: the forecasted values starting after the training data.
    :param forecast_confint: the confidence intervals for the forecast.
    :param in_sample_preds: optional, the in-sample predictions matching the training data.
    :param in_sample_confint: optional, the confidence intervals for the in-sample predictions.
    :return: figure
    """

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot the train time series
    sns.lineplot(ax=ax, x=train.index, y=train.values, label='Train', color='skyblue')

    # Plot the forecast time series and the actual
    sns.lineplot(ax=ax, x=test.index, y=test.values, label='Forecast', color='orange')
    sns.lineplot(ax=ax, x=forecast.index, y=forecast.values, label='Forecast', color='orange')

    # Plot the forecast confidence intervals
    ax.fill_between(forecast.index,
                    forecast.values - forecast_confint,
                    forecast.values + forecast_confint,
                    color='orange', alpha=0.3, label='Forecast Conf. Int.')

    # Plot the in-sample predictions if provided
    if in_sample_preds is not None:
        sns.lineplot(ax=ax, x=in_sample_preds.index, y=in_sample_preds.values, label='In-sample Predictions',
                     color='green')

    # Plot the in-sample confidence intervals if provided
    if in_sample_confint is not None:
        ax.fill_between(in_sample_preds.index,
                        in_sample_preds.values - in_sample_confint,
                        in_sample_preds.values + in_sample_confint,
                        color='green', alpha=0.3, label='In-sample Conf. Int.')

    # Customize the x-axis to show one tick per year
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='lower'))
    ax.set_xticks(train.index.to_timestamp()[train.index.to_timestamp().month == 1])

    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.set_title('Predictions')
    fig.legend()

    return fig


def get_diagnostics_df(
        fit_results: ets.ETSResults | ARIMAResults
) -> pd.DataFrame:
    # df = pd.DataFrame.from_records([{
    #     # Is AIC ever better than AICc?
    #     # https://stats.stackexchange.com/questions/319769/is-aicc-ever-worse-than-aic
    #     "AICc": fit_results.aicc,
    #     "BIC": fit_results.bic
    # }])

    # TODO maybe this is good enough
    return fit_results.summary()


def get_forecast_error_df(test: pd.Series, forecast: pd.Series) -> pd.DataFrame:
    return pd.DataFrame.from_records([{
        "MAE": sk_metrics.mean_absolute_error(test, forecast),
        "RMSE": sk_metrics.root_mean_squared_error(test, forecast),
        "MAPE": sk_metrics.mean_absolute_percentage_error(test, forecast)
    }])


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
