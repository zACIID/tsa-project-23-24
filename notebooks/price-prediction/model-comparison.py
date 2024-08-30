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
import sklearn.model_selection as sk_modsel
import sklearn.metrics as sk_metrics
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tsa.forecasting.stl import STLForecast, STLForecastResults
from pandas.plotting import autocorrelation_plot, lag_plot

import src.visualization as vis
import src.eda as eda
import src.diagnostics as diag
import src.models as mod
# -

# ## Load Data
#
# The data used for prediction will be 95% of the log time series.
# 1. The remaining 5% is the holdout set, to try with the final model, after the selection process
# 2. The log time series has been chosen because differencing/integration will be incorporated by the models

smh = eda.SMH(differencing_periods=1)
smh.load_data(fraction=0.95, left_side=True)

# +
ts: pd.Series = smh.log
train, val = pm_modsel.train_test_split(ts, train_size=0.9)

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
    auto_arima_res: pm.ARIMA = pm.auto_arima(
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
auto_arima_model: SARIMAXResults = auto_arima_res.arima_res_

# `#todo` comment on the results here; see https://analyzingalpha.com/interpret-arima-results#Ljung-box
# - LjungBox seems satisfying (or is it not? i see probQ = 0.94 -> it is not, means p-value is 0.94. Significant Autocorr left at lag 6 and 8), 
# - heteroskedasticity too (means I have captured all the variance right?)

# sigma2 is the estimate of the variance of the error term
auto_arima_model.summary()

_ = auto_arima_model.plot_diagnostics(figsize=(16, 16))

_ = diag.plot_predictions_from_fit_results(
    fit_results=auto_arima_model,
    train=train,
    test=val,
    alpha=0.05,
    differencing_order=1
)

preds, error_df = mod.cv_forecast(
    model=auto_arima_res,  # Need to pass this because it needs get_params method - as last resort I can monkey_patch it to add the method
    ts=ts,
    start_at=train.shape[0],
    step=5,
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
    zoom=0.075
)

importlib.reload(diag)
fit_stats_df = diag.get_diagnostics_df(auto_arima_model)

model_perf = pd.concat([
    model_perf,
    pd.concat([
        pd.DataFrame.from_records([{"model": "auto.arima",}]),
        error_df,
        fit_stats_df
    ], axis=1)
], axis=0)
model_perf

# ## ETS Models



# ## Model Comparisons


