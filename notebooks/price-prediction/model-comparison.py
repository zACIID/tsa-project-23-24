# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # SMH Price Prediction - Model Comparison

# +
import importlib

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
import statsmodels.tsa.exponential_smoothing.ets as ets
import sklearn.model_selection as sk_modsel
import sklearn.metrics as sk_metrics
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tsa.forecasting.stl import STLForecast, STLForecastResults
from pandas.plotting import autocorrelation_plot, lag_plot

import src.visualization as vis
import src.data as data
import src.diagnostics as diag
import src.models as mod
# -

# ## Load Data
#
# The data used for prediction will be split into train and val set in 90-10 fashion, the latter of which will be used to evaluate forecasts.
# The log time series has been chosen because differencing/integration will be incorporated by the models.

smh = data.SMH(differencing_periods=1)
smh.load_data(fraction=1.0, left_side=True)

# +
ts: pd.Series = smh.log
train, val = pm_modsel.train_test_split(ts, train_size=0.95)

# This is so the confint dataframes have two columns: "lower y" and "upper y"
train.name = "y"
val.name = "y"

print(f"Number of training samples: {train.shape[0]}")
print(f"Number of validation samples: {val.shape[0]}")
# -

model_perf = pd.DataFrame()

# ## Comparison Criteria
#
# Models will be compared on their CV RMSE, MAE and MAPE, as well as their AIC and BIC statistics, calculated during fit.
# CV errors are based on a rolling forecast with `horizon=10`, i.e. 2 trading weeks.

# ## SARIMA Models

# ### Auto-ARIMA
#
# One of the models will be the choice of the `auto_arima` (Python's equivalent of R's `auto.arima`) function, which will try to fit SARIMA models without the seasonal component, i.e. ARIMA models.
# The reason for this is that the EDA phase has shown poor evidence of seasonality, which substantially increases running times and uncertainty if included to model when it isn't the case.

with pm.StepwiseContext(max_steps=50):
    auto_arima_run: pm.ARIMA = pm.auto_arima(
        y=train,
        start_p=0, d=1, start_q=0, max_d=1,
        max_order=None,
        seasonal=False,
        stationary=False,
        error_action="trace", trace=True,
        stepwise=True,
        information_criterion="aicc",
        sarimax_kwargs={}
    )
auto_arima_res: SARIMAXResults = auto_arima_run.arima_res_

# sigma2 is the estimate of the variance of the error term
auto_arima_res.summary()

_ = auto_arima_res.plot_diagnostics(figsize=(16, 16))

_ = diag.plot_predictions_from_fit_results(
    fit_results=auto_arima_res,
    train=train,
    test=val,
    alpha=0.05,
    start_at=1
)

preds, error_df = mod.cv_forecast(
    model=auto_arima_run,  # Need to pass this because it needs get_params method - as last resort I can monkey_patch it to add the method
    ts=ts,
    start_at=train.shape[0],
    step=10,
    horizon=10
)

_ = diag.plot_predictions(
    train=train,
    test=val[-(preds.shape[0]):],
    forecast=preds,
    zoom=1.0
)

_ = diag.plot_predictions(
    train=train,
    test=val[-(preds.shape[0]):],
    forecast=preds,
    zoom=0.125
)

fit_stats_df = diag.get_diagnostics_df(auto_arima_res)

model_perf = pd.concat([
    model_perf,
    pd.concat([
        pd.DataFrame.from_records([{"model": "auto.arima",}]),
        error_df,
        fit_stats_df
    ], axis=1)
], axis=0)
model_perf


# ### ARIMA(6, 1, 6)
#

arima_6_1_6 = pm.ARIMA(order=(6, 1, 6), maxiter=25)
arima_6_1_6_res: SARIMAXResults = arima_6_1_6.fit(y=train).arima_res_

# sigma2 is the estimate of the variance of the error term
arima_6_1_6_res.summary()

_ = arima_6_1_6_res.plot_diagnostics(figsize=(16, 16))

_ = diag.plot_predictions_from_fit_results(
    fit_results=arima_6_1_6_res,
    train=train,
    test=val,
    alpha=0.05,
    start_at=8+1,  # seasonality+seasonal differencing
    zoom=1.0
)

_ = diag.plot_predictions_from_fit_results(
    fit_results=arima_6_1_6_res,
    train=train,
    test=val,
    alpha=0.05,
    start_at=8+1,  # seasonality+seasonal differencing
    zoom=0.125
)

preds, error_df = mod.cv_forecast(
    model=arima_6_1_6,  # Need to pass this because it needs get_params method - as last resort I can monkey_patch it to add the method
    ts=ts,
    start_at=train.shape[0],
    step=10,
    horizon=10
)

_ = diag.plot_predictions(
    train=train,
    test=val[-(preds.shape[0]):],
    forecast=preds,
    zoom=1.0
)

fig = diag.plot_predictions(
    train=train,
    test=val[-(preds.shape[0]):],
    forecast=preds,
    zoom=0.05
)
_ = fig.axes[0].set_xticks(list(np.arange(fig.axes[0].get_xlim()[0], fig.axes[0].get_xlim()[1]+1)))

fit_stats_df = diag.get_diagnostics_df(arima_6_1_6_res)

model_perf = pd.concat([
    model_perf,
    pd.concat([
        pd.DataFrame.from_records([{"model": "ARIMA(6,1,6)"}]),
        error_df,
        fit_stats_df
    ], axis=1)
], axis=0)
model_perf



# ### ARIMA(8, 1, 8)

arima_8_1_8 = pm.ARIMA(order=(8, 1, 8), maxiter=25)
arima_8_1_8_res: SARIMAXResults = arima_8_1_8.fit(y=train).arima_res_

# sigma2 is the estimate of the variance of the error term
arima_8_1_8_res.summary()

_ = arima_8_1_8_res.plot_diagnostics(figsize=(16, 16))

_ = diag.plot_predictions_from_fit_results(
    fit_results=arima_8_1_8_res,
    train=train,
    test=val,
    alpha=0.05,
    start_at=8+1,  # seasonality+seasonal differencing
    zoom=1.0
)

_ = diag.plot_predictions_from_fit_results(
    fit_results=arima_8_1_8_res,
    train=train,
    test=val,
    alpha=0.05,
    start_at=8+1,  # seasonality+seasonal differencing
    zoom=0.125
)

preds, error_df = mod.cv_forecast(
    model=arima_8_1_8,  # Need to pass this because it needs get_params method - as last resort I can monkey_patch it to add the method
    ts=ts,
    start_at=train.shape[0],
    step=10,
    horizon=10
)

_ = diag.plot_predictions(
    train=train,
    test=val[-(preds.shape[0]):],
    forecast=preds,
    zoom=1.0
)

fig = diag.plot_predictions(
    train=train,
    test=val[-(preds.shape[0]):],
    forecast=preds,
    zoom=0.05
)
_ = fig.axes[0].set_xticks(list(np.arange(fig.axes[0].get_xlim()[0], fig.axes[0].get_xlim()[1]+1)))

fit_stats_df = diag.get_diagnostics_df(arima_8_1_8_res)

model_perf = pd.concat([
    model_perf,
    pd.concat([
        pd.DataFrame.from_records([{"model": "ARIMA(8,1,8)"}]),
        error_df,
        fit_stats_df
    ], axis=1)
], axis=0)
model_perf


# ## ETS Models


# ### ETS (A, A, None)

ets_a_a_none_params = {
    "error": "add",
    "trend": "add",
    "seasonal": None
}
ets_a_a_none = ets.ETSModel(
    train,
    **ets_a_a_none_params
)
ets_a_a_none_res: ets.ETSResults = ets_a_a_none.fit()

# sigma2 is the estimate of the variance of the error term
ets_a_a_none_res.summary()

_ = diag.residuals_diagnostics(ets_a_a_none_res.fittedvalues, ets_a_a_none_res.resid)

_ = diag.plot_predictions_from_fit_results(
    fit_results=ets_a_a_none_res,
    train=train,
    test=val,
    alpha=0.05,
    start_at=0,
    zoom=1.0
)

_ = diag.plot_predictions_from_fit_results(
    fit_results=ets_a_a_none_res,
    train=train,
    test=val,
    alpha=0.05,
    start_at=0,
    zoom=0.125
)

preds, error_df = mod.cv_forecast(
    # Same params as above, just without the endog which does not need to be specified
    model=mod.ETSModelEstimatorWrapper(ets_a_a_none_params),
    ts=ts,
    start_at=train.shape[0],
    step=10,
    horizon=10
)

_ = diag.plot_predictions(
    train=train,
    test=val[-(preds.shape[0]):],
    forecast=preds,
    zoom=1.0
)

fig = diag.plot_predictions(
    train=train,
    test=val[-(preds.shape[0]):],
    forecast=preds,
    zoom=0.05
)
_ = fig.axes[0].set_xticks(list(np.arange(fig.axes[0].get_xlim()[0], fig.axes[0].get_xlim()[1]+1)))

fit_stats_df = diag.get_diagnostics_df(ets_a_a_none_res)

model_perf = pd.concat([
    model_perf,
    pd.concat([
        pd.DataFrame.from_records([{"model": "ETS(A, A, None)"}]),
        error_df,
        fit_stats_df
    ], axis=1)
], axis=0)
model_perf


# ### ETS (A, A_d, None)

ets_a_ad_none_params = {
    "error": "add",
    "trend": "add",
    "damped_trend": True,
    "seasonal": None
}
ets_a_ad_none = ets.ETSModel(
    train,
    **ets_a_ad_none_params
)
ets_a_ad_none_res: ets.ETSResults = ets_a_ad_none.fit()

# sigma2 is the estimate of the variance of the error term
ets_a_ad_none_res.summary()

_ = diag.residuals_diagnostics(ets_a_ad_none_res.fittedvalues, ets_a_ad_none_res.resid)

_ = diag.plot_predictions_from_fit_results(
    fit_results=ets_a_ad_none_res,
    train=train,
    test=val,
    alpha=0.05,
    start_at=0,
    zoom=1.0
)

_ = diag.plot_predictions_from_fit_results(
    fit_results=ets_a_ad_none_res,
    train=train,
    test=val,
    alpha=0.05,
    start_at=0,
    zoom=0.125
)

preds, error_df = mod.cv_forecast(
    # Same params as above, just without the endog which does not need to be specified
    model=mod.ETSModelEstimatorWrapper(ets_a_ad_none_params),
    ts=ts,
    start_at=train.shape[0],
    step=10,
    horizon=10
)

_ = diag.plot_predictions(
    train=train,
    test=val[-(preds.shape[0]):],
    forecast=preds,
    zoom=1.0
)

fig = diag.plot_predictions(
    train=train,
    test=val[-(preds.shape[0]):],
    forecast=preds,
    zoom=0.05
)
_ = fig.axes[0].set_xticks(list(np.arange(fig.axes[0].get_xlim()[0], fig.axes[0].get_xlim()[1]+1)))

fit_stats_df = diag.get_diagnostics_df(ets_a_ad_none_res)

model_perf = pd.concat([
    model_perf,
    pd.concat([
        pd.DataFrame.from_records([{"model": "ETS(A, A_d, None)"}]),
        error_df,
        fit_stats_df
    ], axis=1)
], axis=0)
model_perf



# ### ETS (A, A_d, A)

ets_a_ad_a_params = {
    "error": "add",
    "trend": "add",
    "damped_trend": True,
    "seasonal": "add",
    "seasonal_periods": 8
}
ets_a_ad_a = ets.ETSModel(
    train,
    **ets_a_ad_a_params
)
ets_a_ad_a_res: ets.ETSResults = ets_a_ad_a.fit()


# sigma2 is the estimate of the variance of the error term
ets_a_ad_a_res.summary()

_ = diag.residuals_diagnostics(ets_a_ad_a_res.fittedvalues, ets_a_ad_a_res.resid)

_ = diag.plot_predictions_from_fit_results(
    fit_results=ets_a_ad_a_res,
    train=train,
    test=val,
    alpha=0.05,
    start_at=8+1,  # seasonality+seasonal differencing
    zoom=1.0
)

_ = diag.plot_predictions_from_fit_results(
    fit_results=ets_a_ad_a_res,
    train=train,
    test=val,
    alpha=0.05,
    start_at=8+1,  # seasonality+seasonal differencing
    zoom=0.125
)

preds, error_df = mod.cv_forecast(
    # Same params as above, just without the endog which does not need to be specified
    model=mod.ETSModelEstimatorWrapper(ets_a_ad_a_params),
    ts=ts,
    start_at=train.shape[0],
    step=10,
    horizon=10
)

_ = diag.plot_predictions(
    train=train,
    test=val[-(preds.shape[0]):],
    forecast=preds,
    zoom=1.0
)

fig = diag.plot_predictions(
    train=train,
    test=val[-(preds.shape[0]):],
    forecast=preds,
    zoom=0.05
)
_ = fig.axes[0].set_xticks(list(np.arange(fig.axes[0].get_xlim()[0], fig.axes[0].get_xlim()[1]+1)))

fit_stats_df = diag.get_diagnostics_df(ets_a_ad_a_res)

model_perf = pd.concat([
    model_perf,
    pd.concat([
        pd.DataFrame.from_records([{"model": "ETS(A, A_d, A)"}]),
        error_df,
        fit_stats_df
    ], axis=1)
], axis=0)
model_perf




# ## Model Comparisons

model_perf
