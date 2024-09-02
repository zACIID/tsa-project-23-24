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

# # SMH Price Prediction

# +
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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.forecasting.stl import STLForecast, STLForecastResults
from pandas.plotting import autocorrelation_plot, lag_plot

import src.visualization as vis
import src.data as data
import src.diagnostics as diag
import src.transfer_function_modelling as tf_mod
# -

# ## SMH and SemiconductorSales Summary

smh = data.SMHForSemiconductorSales(differencing_periods=1)
smh.load_data(fraction=0.85, left_side=True)

sales = data.SemiconductorSales(differencing_periods=1)
sales.load_data(fraction=0.85, left_side=True)

# These two will be useful to make predictions later

smh_full = data.SMHForSemiconductorSales(differencing_periods=1)
smh_full.load_data(fraction=1.0, left_side=True)

sales_full = data.SemiconductorSales(differencing_periods=1)
sales_full.load_data(fraction=1.0, left_side=True)

# ## Desesonalizing

decompose_result = tsa_season.seasonal_decompose(
    sales.log_diffed,
    period=12,
    two_sided=False,
)
sales.log_diffed = (sales.log_diffed - decompose_result.seasonal).dropna()

decompose_result = tsa_season.seasonal_decompose(
    sales_full.log_diffed,
    period=12,
    two_sided=False,
)
sales_full.log_diffed = (sales_full.log_diffed - decompose_result.seasonal).dropna()

# ### ACCF Grid Plots

_ = vis.plot_accf_grid(x=sales.original, y=smh.original, x_name="Sales", y_name="SMH Price")

_ = vis.plot_accf_grid(x=sales.diffed, y=smh.diffed, x_name="Sales", y_name="SMH Price")

_ = vis.plot_accf_grid(x=sales.log, y=smh.log, x_name="Sales", y_name="SMH Price")

_ = vis.plot_accf_grid(x=sales.log_diffed, y=smh.log_diffed, x_name="Sales", y_name="SMH Price")


# ### Stationarity Tests

smh.stationarity_tests()

sales.stationarity_tests()

# ## Pre-whitening
#
# Below the most promising pre-whitened pairs of (predictors, target) are shown

# ### Log-diffed

log_diffed_pw = tf_mod.PreWhitening(
    x=sales.log_diffed, y=smh.log_diffed, x_name="Sales", y_name="SMH Price"
)
log_diffed_pw.prewhiten(x_differencing_order=0)

log_diffed_pw.plot_visualizations()

log_diffed_pw.model_x.summary()

# ## Transfer Function

model_res, preds = tf_mod.sequential_tf_model_predictions(
    prewhitened_ts=log_diffed_pw,
    exog_series=sales_full.log_diffed,
    lags=13,
    endog_series_for_plot=smh_full.log_diffed,
    plot_predictions=True
)

model_res.summary()

preds.summary_frame()

# ## Lagged Regression - AutoReg Model
#
# I am referring to the pattern shown [here](https://online.stat.psu.edu/stat510/lesson/9/9.1#paragraph--309) to choose the terms of my lagged regression model.
# I am trying multiple patterns because it is unclear under which my case falls.

# ### Pattern 1
#
# [pattern 1](https://online.stat.psu.edu/stat510/lesson/9/9.1#paragraph--309): $d$ here is 5 because it is the first lag that is close to the significance bands, no lags for $y$. Additional lags of x that are close to the bands are 10 and 8.

import importlib
importlib.reload(tf_mod)

x_lags = [5, 8, 10]
y_lags = 0

lagged_model_res, lagged_model_preds = tf_mod.lagged_regression_model_predictions(
    x_lags=x_lags,
    y_lags=y_lags,
    endog=smh_full.log_diffed,
    exog=sales_full.log_diffed,
    train_fraction=0.85,
    plot_predictions=True
)

lagged_model_res.summary()

lagged_model_res.aicc

lagged_model_preds.summary_frame()


# ### Pattern 4
#
# [pattern 4](https://online.stat.psu.edu/stat510/lesson/9/9.1#paragraph--309): $d$ here is 5 because the first that is close to the significance bands, lag 1 and 2 for $y$

import importlib
importlib.reload(tf_mod)

x_lags = [12]
y_lags = 2

lagged_model_res, lagged_model_preds = tf_mod.lagged_regression_model_predictions(
    x_lags=x_lags,
    y_lags=y_lags,
    endog=smh_full.log_diffed,
    exog=sales_full.log_diffed,
    train_fraction=0.85,
    plot_predictions=True
)

lagged_model_res.summary()

lagged_model_res.aicc

lagged_model_preds.summary_frame()


# ### Pattern 5
#
# [pattern 5](https://online.stat.psu.edu/stat510/lesson/9/9.1#paragraph--309): lags 5, 8, 10 for $x$ because they are the first that come close to the significance bands, lag 1 and 2 for $y$

import importlib
importlib.reload(tf_mod)

x_lags = [12, 13]
y_lags = 2

lagged_model_res, lagged_model_preds = tf_mod.lagged_regression_model_predictions(
    x_lags=x_lags,
    y_lags=y_lags,
    endog=smh_full.log_diffed,
    exog=sales_full.log_diffed,
    train_fraction=0.85,
    plot_predictions=True
)

lagged_model_res.summary()

lagged_model_res.aicc

lagged_model_preds.summary_frame()


