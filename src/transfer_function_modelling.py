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
        self.x_whitened: pd.Series | None = None
        self.y_whitened: pd.Series | None = None

    def prewhiten(self, x_differencing_order: int) -> typing.Tuple[pd.Series, pd.Series]:
        x_whitened, y_whitened = prewhiten(self._x, self._y, x_differencing_order)

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



def prewhiten(x: pd.Series, y: pd.Series, x_differencing_order: int) -> typing.Tuple[pd.Series, pd.Series]:
    """
    :param x: lagged ts, which is ideally used to try to predict y
    :param y:
    :param x_differencing_order: differencing order of x, provided as param to the ARIMA model
    :return:
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

    return x_whitened, y_whitened

