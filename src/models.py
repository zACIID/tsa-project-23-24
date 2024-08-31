import typing

import pandas as pd
import pmdarima.model_selection as pm_modsel
import statsmodels.tsa.exponential_smoothing.ets as ets
import sklearn.base as sk_base

import src.diagnostics as diag


# NOTE: I decided to implemented only these methods because they are the only
#   required by the pm_modsel.cross_val_predict function. I looked at the source code
class ETSModelEstimatorWrapper(sk_base.BaseEstimator):
    def __init__(self, ets_model_params: typing.Dict[str, typing.Any]):
        self._ets: ets.ETSResults | None = None
        self._ets_params = ets_model_params

    def fit(self, y, X, **fit_args):
        self._ets = ets.ETSModel(endog=y, **self._ets_params).fit()

    def fit_predict(self, y, X=None, n_periods=10, **fit_args):
        self.fit(y, X, **fit_args)

        return self.predict(n_periods=n_periods, X=X)

    def predict(self, n_periods, X, return_conf_int=False, alpha=0.05, **kwargs):
        self._ets.predict(start=self._ets.nobs, end=self._ets.nobs + n_periods)



def cv_forecast(
        model: sk_base.BaseEstimator,
        ts: pd.Series,
        start_at: int | float = 0.75,
        step: int = 5,
        horizon: int = 5,
        training_window: int = None
) -> typing.Tuple[pd.Series, pd.DataFrame]:
    """
    Perform recursive `horizon`-step-ahead prediction
    :param model: anything that implements the `fit`, `fit_predict` and `predict` sklearn Estimator interface
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
        # test_size = ts.shape[0] - training_window
    else:
        start_at = int(start_at*ts.shape[0]) if isinstance(start_at, float) else start_at
        cv = pm_modsel.RollingForecastCV(step=step, h=horizon, initial=start_at)
        # test_size = ts.shape[0] - start_at

    # TODO not sure what preds actually is here, may not be a pd.Series and hence throw an error with get_forecast_error_df below
    preds = pm_modsel.cross_val_predict(
        model,
        y=ts,
        cv=cv,
        verbose=2
    )

    # Apparently preds is a np.ndarray with fewer samples than test_size
    test_ts = ts[-(preds.shape[0]):]
    preds = pd.Series(preds, index=test_ts.index)
    error_df = diag.get_forecast_error_df(test=test_ts, forecast=preds)

    return preds, error_df
