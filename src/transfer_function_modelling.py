import typing

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
from statsmodels.tsa.ar_model import AutoReg, AutoRegResults
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.forecasting.stl import STLForecast, STLForecastResults
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from pandas.plotting import autocorrelation_plot, lag_plot

import src.visualization as vis
import src.data as data
import src.diagnostics as diag


class PreWhitening:
    def __init__(self, x: pd.Series, y: pd.Series, x_name: str = "X", y_name: str = "Y"):
        self._x: pd.Series = x
        self._x_name: str = x_name
        self._y: pd.Series = y
        self._y_name: str = y_name

        self.model_x: SARIMAXResults | None = None
        """Model fitted on X which was used to whiten both series. Set only after having called prewhiten()"""

        self.x_whitened: pd.Series | None = None
        self.y_whitened: pd.Series | None = None

    def prewhiten(self, x_differencing_order: int) -> typing.Tuple[pd.Series, pd.Series]:
        """
        :param x_differencing_order:
        :return: (x_whitened, y_whitened)
        """

        model_x, x_whitened, y_whitened = prewhiten(self._x, self._y, x_differencing_order)

        self.model_x = model_x

        # Remove first `differencing_order` samples because can't make predictions for those
        self.x_whitened, self.y_whitened = x_whitened[x_differencing_order:], y_whitened[x_differencing_order:]

        return x_whitened, y_whitened

    def plot_visualizations(self):
        vis.ts_eda(
            self.x_whitened,
            ts_name=f"{self._x_name}_{{whitened}}",
            expected_seasonality=1,
            ts_plot=False,
            acf_pacf_plots=False,
            top_k_autocorr_plots=False,
            ts_decomposition=False,
            season_boxplots=False
        )
        vis.ts_eda(
            self.y_whitened,
            ts_name=f"{self._y_name}_{{whitened}}",
            expected_seasonality=1,
            ts_plot=False,
            acf_pacf_plots=False,
            top_k_autocorr_plots=False,
            ts_decomposition=False,
            season_boxplots=False
        )

        _ = vis.plot_accf_grid(
            x=self.x_whitened, y=self.y_whitened,
            x_name=f"{self._x_name}_{{whitened}}", y_name=f"{self._y_name}_{{whitened}}",
            expected_seasonality=12,  # Even if no expected season, this param determines the number of lags shown
        )
        _ = vis.plot_top_k_ccf_lags(x=self.x_whitened, y=self.y_whitened, k=10, max_lag=50)



def prewhiten(x: pd.Series, y: pd.Series, x_differencing_order: int) -> typing.Tuple[SARIMAXResults, pd.Series, pd.Series]:
    """
    :param x: lagged ts, which is ideally used to try to predict y
    :param y:
    :param x_differencing_order: differencing order of x, provided as param to the ARIMA model
    :return: (whitening model fitted on x, x_whitened, y_whitened)
    """

    with pm.StepwiseContext(max_steps=50):
        auto_arima_run: pm.ARIMA = pm.auto_arima(
            y=x,
            start_p=0, d=x_differencing_order, start_q=0, max_d=x_differencing_order,
            max_order=None,
            seasonal=False,
            error_action="trace", trace=True,
            stepwise=True,
            information_criterion="aicc",
            sarimax_kwargs={}
        )
    model: SARIMAXResults = auto_arima_run.arima_res_

    # Step 2: Filter (whiten) both X and Y using the fitted AR model coefficients
    x_whitened = model.resid

    # Check this answer: https://stackoverflow.com/a/66996325/19582401
    # arma_innovations with ARIMA model params gives exactly the residuals
    #   of the model on the provided time series
    y_whitened = arma_innovations(
        y,

        # TODO don't understand the minus part on the AR coefficients
        # Skip the first param because it is the intercept
        ar_params=-model.polynomial_reduced_ar[1:],
        ma_params=model.polynomial_reduced_ma[1:]
    )[0]

    x_whitened = pd.Series(x_whitened, index=x.index)
    y_whitened = pd.Series(y_whitened, index=y.index)

    return model, x_whitened, y_whitened


def sequential_tf_model_predictions(
        prewhitened_ts: PreWhitening,
        exog_series: pd.Series,
        lags: int = 5,
        endog_series_for_plot: pd.Series | None = None,
        plot_predictions: bool = False
) -> typing.Tuple[AutoRegResults, PredictionResults]:
    """
    Estimates the coefficients alpha of an AutoReg model based on the
        prewhitened series and the model used to calculate them.
    This is the "sequential methodology" illustrated in the course",
        the R implementation of which is provided below.
    Makes prediction with said model with the provided exog data.

    ```R
    lag.max<-5 # edit it if you need more
    ccov<-ccf(r.Quotes,r.TV.advert,  type =
                c("covariance"),plot=FALSE,lag.max = lag.max)
    alpha.hat<-ccov$acf[(lag.max+1):(2*lag.max+1)]/fit.x$sigma2

    alpha.hat
    ```

    :param prewhitened_ts:
    :param exog_series: series to use to extract lagged exogenous predictors x_{t}, x_{t-1}, ..., x_{t-lags}.
        The number of in_sample and out_of_sample points is determined by the length of `prewhitened_ts.y_whitened`,
            meaning that this series must be longer than that.
    :param lags: number of lags to use to estimate parameters.
        The final number of parameters alpha_i is equal to lags
    :param endog_series_for_plot: Series whose in_sample fraction is the original
        version of `prewhitened_ts.y_whitened`. It must be as long as exog_series.
        Only required in case of plotting, as the model doesn't need to be fit with this series.
    :param plot_predictions: whether to also plot the predictions of the model,
        both in and out of sample. Must also provide endog_series if this is True
    :return: (model, prediction results)
    """

    # NOTE old comments
    #     :param in_sample_exog: this should be a numpy matrix of shape (nobs, k) containing
    #         k consecutive lags from the original version of the x_whitened series.
    #         They should be placed in such a way that y[i] is regressed onto x[i, 1], x[i, 2], ...
    #     :param oos_exog: out-of-sample exog, i.e. a matrix of the same shape as exog,
    #         with the exog vars to use on out of sample predictions. The shape of this
    #         also determines how long should be the forecast, i.e. the out of sample predicted points.

    # x and y are swapped so that positive lags are applied to y
    ccov = tsa.ccovf(y=prewhitened_ts.x_whitened, x=prewhitened_ts.y_whitened)

    # TODO DEBUG
    print(f"Cross covariance up to lag {lags}:")
    print(ccov[:lags])

    # Last param is sigma2
    sigma2 = prewhitened_ts.model_x.arparams[-1]
    alpha = ccov[:lags] / sigma2
    x_lags = alpha.shape[0]
    exog_predictors = np.array([
        exog_series.shift(i) for i in range(x_lags)
    ]).T

    # Remove every row that has at least one nan, which should be the first x_lags row
    exog_predictors = exog_predictors[~np.isnan(exog_predictors).any(axis=1), :]

    # Take the latest values to account for the removed lags which didn't have all X predictors
    # It is not important what the endog is because we are not actually fitting the model,
    #   but it must have matching shape with in_sample_exog so that no error is raised
    endog = np.ones((prewhitened_ts.y_whitened.shape[0]))[-(exog_predictors.shape[0]):]
    in_sample_exog = exog_predictors[:endog.shape[0], :]
    oos_exog = exog_predictors[endog.shape[0]:, :]

    sequential_tf: AutoReg = AutoReg(endog=endog, lags=0, exog=in_sample_exog)

    # This is basically the intercept. Since the sequential method doesn't estimate it, we set it to 0.
    # AutoReg always adds a "const" param on top of the x-predictors/ar-terms, which we want to set to 0
    alpha = np.insert(alpha, 0, 0, axis=0)
    sequential_tf_res: AutoRegResults = AutoRegResults(
        model=sequential_tf,
        params=alpha,

        # Fake cov matrix, do not need that to call predict() or get_predictions
        cov_params=np.ones((alpha.shape[0], alpha.shape[0]))
    )

    # Get in and out of sample predictions
    pred_res = sequential_tf_res.get_prediction(
        start=0,
        end=in_sample_exog.shape[0] + oos_exog.shape[0] - 1,  # because start is 0-indexed
        exog_oos=oos_exog
    )

    if plot_predictions:
        if endog_series_for_plot is None:
            raise ValueError("Since plot_predictions is True, endog_series must be provided")

        endog_series_for_plot = endog_series_for_plot[-(in_sample_exog.shape[0] + oos_exog.shape[0]):]  # Account for lags
        preds_df = pred_res.summary_frame()
        in_sample_nobs = in_sample_exog.shape[0]
        diag.plot_predictions(
            train=endog_series_for_plot[:in_sample_nobs],
            test=endog_series_for_plot[in_sample_nobs:],
            forecast=preds_df["mean"][in_sample_nobs:],
            forecast_confint=(preds_df["mean_ci_lower"][in_sample_nobs:], preds_df["mean_ci_upper"][in_sample_nobs:]),
            in_sample_preds=preds_df["mean"][:in_sample_nobs],
            in_sample_confint=(preds_df["mean_ci_lower"][:in_sample_nobs], preds_df["mean_ci_upper"][:in_sample_nobs]),
        )
        diag.plot_predictions(
            train=endog_series_for_plot[:in_sample_nobs],
            test=endog_series_for_plot[in_sample_nobs:],
            forecast=preds_df["mean"][in_sample_nobs:],
            forecast_confint=(preds_df["mean_ci_lower"][in_sample_nobs:], preds_df["mean_ci_upper"][in_sample_nobs:]),
            in_sample_preds=preds_df["mean"][:in_sample_nobs],
            in_sample_confint=(preds_df["mean_ci_lower"][:in_sample_nobs], preds_df["mean_ci_upper"][:in_sample_nobs]),
            zoom=1 - ((in_sample_nobs - 10) / exog_predictors.shape[0]),  # Show the last 10 in-sample + all OOS
        )

    return sequential_tf_res, pred_res


def lagged_regression_model_predictions(
        x_lags: typing.Sequence[int],
        y_lags: int,
        endog: pd.Series,
        exog: pd.Series,
        train_fraction: float = 0.95,
        plot_predictions: bool = False
) -> typing.Tuple[AutoRegResults, PredictionResults]:
    """
    :param x_lags: collection of lags to apply to the x (exog) variable
    :param y_lags: autoregressive order of y
    :param endog: target/dependent variable y
    :param exog: predictor x
    :param train_fraction: number of samples to use to fit the model;
        the remaining will be used as test set and for out-of-sample forecast
    :param plot_predictions:
    :return:
    """

    exog_predictors = np.array([
        exog.shift(i) for i in x_lags
    ]).T

    # Remove every row that has at least one nan, which should be the first x_lags row
    exog_predictors = exog_predictors[~np.isnan(exog_predictors).any(axis=1), :]

    # Take the latest values to account for the removed lags which didn't have all X predictors
    endog = endog[-(exog_predictors.shape[0]):]

    in_sample_nobs = int(endog.shape[0]*train_fraction)
    in_sample_endog, oos_endog = endog[:in_sample_nobs], endog[in_sample_nobs:]
    in_sample_exog, oos_exog = exog_predictors[:in_sample_nobs, :], exog_predictors[in_sample_nobs:, :]

    lagged_reg_model_res: AutoRegResults = AutoReg(endog=in_sample_endog, lags=y_lags, exog=in_sample_exog).fit()
    pred_res = lagged_reg_model_res.get_prediction(
        start=0,
        end=in_sample_endog.shape[0] + oos_endog.shape[0] - 1,  # because start is 0-indexed
        exog_oos=oos_exog
    )

    if plot_predictions:
        preds_df = pred_res.summary_frame()
        in_sample_nobs = in_sample_exog.shape[0]
        diag.plot_predictions(
            train=in_sample_endog,
            test=oos_endog,
            forecast=preds_df["mean"][in_sample_nobs:],
            forecast_confint=(preds_df["mean_ci_lower"][in_sample_nobs:], preds_df["mean_ci_upper"][in_sample_nobs:]),
            in_sample_preds=preds_df["mean"][:in_sample_nobs],
            in_sample_confint=(preds_df["mean_ci_lower"][:in_sample_nobs], preds_df["mean_ci_upper"][:in_sample_nobs]),
        )
        diag.plot_predictions(
            train=in_sample_endog,
            test=oos_endog,
            forecast=preds_df["mean"][in_sample_nobs:],
            forecast_confint=(preds_df["mean_ci_lower"][in_sample_nobs:], preds_df["mean_ci_upper"][in_sample_nobs:]),
            in_sample_preds=preds_df["mean"][:in_sample_nobs],
            in_sample_confint=(preds_df["mean_ci_lower"][:in_sample_nobs], preds_df["mean_ci_upper"][:in_sample_nobs]),
            zoom=1 - ((in_sample_nobs - 10) / endog.shape[0]),  # Show the last 10 in-sample + all OOS
        )

    return lagged_reg_model_res, pred_res
