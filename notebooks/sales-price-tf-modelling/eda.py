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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.forecasting.stl import STLForecast, STLForecastResults
from pandas.plotting import autocorrelation_plot, lag_plot

import src.visualization as vis
import src.data as data
import src.diagnostics as diag
import src.transfer_function_modelling as tf_mod
# -

# ## SMH EDA
#

smh = data.SMHForSemiconductorSales(differencing_periods=1)
smh.load_data(fraction=0.95, left_side=True)

# ### Original TS

smh.plot_visualizations(original=True)

# ### 1-Differenced TS

smh.plot_visualizations(diffed=True)

# ### Log-transformed TS

smh.plot_visualizations(log=True)

# ### 1-Differenced Log-transformed TS

smh.plot_visualizations(log_diffed=True)


# ### Stationarity Tests

smh.stationarity_tests()

# ## Semiconductor Sales EDA

sales = data.SemiconductorSales(differencing_periods=1)
sales.load_data(fraction=0.95, left_side=True)

# ### Original TS

sales.expected_seasonality = 48
sales.plot_visualizations(original=True)

# ### 1-Differenced TS

sales.expected_seasonality = 12
sales.plot_visualizations(diffed=True)

# ### Log-transformed TS

sales.expected_seasonality = 48
sales.plot_visualizations(log=True)

# ### 1-Differenced Log-transformed TS

sales.expected_seasonality = 12
sales.plot_visualizations(log_diffed=True)


# ### Stationarity Tests

sales.stationarity_tests()

# ## ACCF Grid Plots

_ = vis.plot_accf_grid(x=sales.original, y=smh.original, x_name="Sales", y_name="SMH Price")

_ = vis.plot_accf_grid(x=sales.diffed, y=smh.diffed, x_name="Sales", y_name="SMH Price")

_ = vis.plot_accf_grid(x=sales.log, y=smh.log, x_name="Sales", y_name="SMH Price")

_ = vis.plot_accf_grid(x=sales.log_diffed, y=smh.log_diffed, x_name="Sales", y_name="SMH Price")
