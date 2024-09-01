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
# -

# ## EDA
#
# Performing EDA only on the train split, i.e. the first 95% of the TS.
# Let the holdout set be a very small percentage of the total data, we care to use as much of the TS as possible for train and validation,
# like in the SIL project. Besides, my forecast horizon is very short, meaning that I do not need to reserve many data points.

smh = data.SMH(differencing_periods=1)
smh.load_data(fraction=0.95, left_side=True)

# ### Number of Data Points Per Year

smh.years_cardinality()

# ### Original TS

smh.plot_visualizations(original=True)

# ### 1-Differenced TS

smh.plot_visualizations(diffed=True)

# ### Log-transformed TS

smh.plot_visualizations(log=True)

# ### 1-Differenced Log-transformed TS

smh.plot_visualizations(log_diffed=True)


