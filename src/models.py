import typing

import pandas as pd
import pmdarima.model_selection as pm_modsel

import src.diagnostics as diag


# def model_fit(
#         ets: bool,
#         model_params: typing.Dict[str, typing.Any]
# ) -> ets.ETSResults | ARIMAResults:
#     """
#     :param ets: TODO could be an enum. If ETS, fit holtwinters, else SARIMAX (else other??)
#     :param model_params:
#     :return: fit_results
#     """
#
#     # TODO UPDATE: this method is probably useless since I just need to call model.fit() to get the results, might as well do it on the notebook
#
#     # TODO
#     #  https://stackoverflow.com/questions/70277316/how-to-take-confidence-interval-of-statsmodels-tsa-holtwinters-exponentialsmooth
#     #  It says:
#     #  1. use ETSModel because of more modern interface
#     #  2. gives me a generalized way to calculate prediction intervals
#     #  Not sure I would use the raw simulate() method though, I would probably go for the get_prediction, which may give me the CI intervals already
#
#     # TODO2: the ETS model has the same interface as ARIMA, meaning that I could just create one method for fit
#     raise NotImplementedError()
#

def cv_forecast(
        model: typing.Any,
        ts: pd.Series,
        start_at: int | float = 0.75,
        step: int = 5,
        horizon: int = 5,
        training_window: int = None
) -> typing.Tuple[pd.Series, pd.DataFrame]:
    """
    Perform recursive `horizon`-step-ahead prediction
    :param model: anything that implements a `fit` method with no required arguments
    :param ts: time series
    :param start_at: either fraction or number of samples - ignored if `training_window` is provided
    :param step: steps to increase the training set size by.
        NOTE: if step = horizon, forecasts from consecutive models have no overlap
    :param horizon: forecast horizon, in number of time steps
    :param training_window: if provided, uses a SlidingWindowForecastCV from the pmdarima library,
        meaning that the model is trained each time on the latest `training_window` points from the time series.
        If not provided, trains on the whole, progressively increasing, time series (see RollingForecastCV from pmdarima)
    :return: tuple of the form (mean predictions, error dataframe from diagnostics.py module -> called on mean_predictions).
        If `training_window` is not provided, then the test set is assumed to be the last (1 - `start_at`) fraction of samples,
        If it is provided, then the test set is everything but the initial training window.
        The error dataframe is calculated only on the test set.
    """

    # TODO check this out:
    #  https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html#Specifying-the-number-of-forecasts
    #  Full on-tutorial for h-step forecasting with time series
    #  Maybe I can do this automatically with cv_predict, but maybe I still need a loop to get the predictions and errors out. Doing it manually maybe gives me more control
    #  I wouldn't worry about CI's here: I first plot the predictions of a single model with a test.length-step-forecast, which gives me a "worst case CI interval",
    #  and then use this technique to just compute errors. This is because averaging CI's I am not so sure about...
    #
    #  I could monte carlo or run multiple simulations and then get the CI using quantiles but it may be very costly...
    #       maybe I can add a parameter here to§

    if training_window is not None:
        cv = pm_modsel.SlidingWindowForecastCV(step=step, h=horizon, window_size=training_window)
        test_size = ts.shape[0] - training_window
    else:
        cv = pm_modsel.RollingForecastCV(step=step, h=horizon, initial=start_at)
        test_size = (1 - start_at)*ts.shape[0]

    # TODO not sure what preds actually is here, may not be a pd.Series and hence throw an error with get_forecast_error_df below
    preds = pm_modsel.cross_val_predict(
        model,
        y=ts,
        cv=cv
    )
    test_ts = ts[-test_size:]
    error_df = diag.get_forecast_error_df(test=test_ts, forecast=preds)

    return preds, error_df





# TODO stuff to do in the notebook:
"""
- SARIMA section
    - fit SARIMA via auto.arima (to choose the best params) and then by hand
    - pass each model to the above two methods to produce fit and cv results
- ETS section
    - fit ETS based on intuition (everything is additive because I am fitting on log, then I check if I want to model seasonality or not, if I want to dampen... etc.
    - pass each model to the above two methods to produce fit and cv results
- compare models based on the errors calculated via the cv_forecast errors and also via the AIC, BIC that can be easily obtained by the fit_results objects
    - produce a final dataframe so that it is easy to look at the table
"""