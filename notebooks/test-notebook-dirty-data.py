# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Test
#
# ## Stuff to do in the project
#
# - Preliminary: I should transform the .xlsx of the semiconductor sales TS into a .csv prior to the analysis, so that it is readily available
# - **First task: stock price prediction**
# 	- *First section: data exploration*
# 		- plot of the smh time series to check them and compare them
# 		- ACF, PACF plots
# 			- 95% confidence bands are valid if we have "a large enough number of points". Is my sample size enough?
# 			- comment the plots to describe how the series seems to behave
# 				- see interpretation of ACF plot at page 24 of tsa notes
# 		- lagplots to understand if the relationship between a point and its lagged version is actually linear, in which case autocorrelation makes sense - else it doesn't, should probably look at Average Mutual Information
# 		- Decompositions:
# 			- Classic and/or STL
# 			- comment on the components (behavior, impact in terms of absolute values, ...)
# 		- other stuff? #todo
# 	- *Engineering*
# 		- Filtering/Smoothing? #todo when should I use filtering/smoothing?
# 		- Try differencing if #todo (if what? there is still autocorrelation left in the noise? I don't remember atm)
# 			- I actually think I should use differencing if the series needs detrending because it is not stationary. How to check for stationarity? various tests (ADF, KPSS) and possibly an analysis of rolling mean and std: they should be independent of time windows (covariance only dependent on lag) and constant for a given lag
# 	- *Modelling*
# 		- Holt-Winters with seasons
# 		- SARIMA
# 			- check the open articles on SARIMA and stock prediction
# 		- which non linear model to try? SETAR? LSTM??? #todo
# 			- especially if non linear relationships are highlighted by lagplots
# - **Second task: Transfer Function Modelling**		
# 	- *First section: data exploration*:
# 		- Create a time series from the SMH price that is in the same scale as that of the semiconductor sales
# 		- *same framework as the data expl. section for the previous task*
# 	- *Second section: engineering*
# 		- prewhitening
# 		- #todo ????
# 	- *Third section: modelling*
# 		- transfer function modelling how?
#
#
# ### Modelling Procedure (from TSA notes)
#
# Modelling procedure
# 1. Plot the data and identify unusual observations that could potentially
# affect the result.
# 2. If necessary use box-cox transformations to reduce the variance.
# 3. If not stationary, take first differencing (moving from ARIMA to ARMA)
# to make the process stationary.
# 4. Examine ACF and PACF to understand AR(p) and MA(q) components.
# 5. Fit chosen models and use IC to select the best one.
# 6. Check residuals using ACF or Portmanteau tests.
# 7. If residuals behave like white noise, calculate forecasts.

# +
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as tsa
import statsmodels.graphics.tsaplots as tsag
import statsmodels.tsa.arima_model as arima
import statsmodels.tsa.seasonal as tsa_season
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pandas.plotting import autocorrelation_plot, lag_plot
# -



# ## SMH Time Series EDA 

# index_col=0 so that timestamp is the index -> handy for plots
smh = pd.read_csv('../data/smh-2000-06-05_2024-07-30__dirty.csv', parse_dates=True, index_col=0)
smh.describe()

smh.head(10)

smh.plot(subplots=True, figsize=(15, 15))

# We are only interested in closing prices

# Just take the Series (i.e. the column) so that it is easier to handle
# Sort on the index (timestamp) so that the order is chronological, 
#   as opposed to having the latest time points come first (for some unknown reason)
smh_close_prices = smh["close"].sort_index(ascending=True)

smh_close_prices.plot(figsize=(12, 8))

# ## Autocorrelation Plots
#
# Note: [How many lags to show in (P)ACF plot](https://stats.stackexchange.com/a/81432/395870)

autocorrelation_plot(smh_close_prices)

# +
fig, ax = plt.subplots(figsize=(12, 8))

# Without the assignment it shows 2 plots for some reason...
fig = tsag.plot_acf(smh_close_prices, ax=ax, lags=100, auto_ylims=True)
# -

# ## TODO stuff to do next:
# - i need to make the series stationary so that I can use SARIMA models. Why is it that I need stationarity for SARIMA models? I do not remember actually...
# - I need to difference the time series until i get to stationarity, i.e. I remove Autocorrelation... careful of overdifferencing. Why is it that autocorr is 0 for stationary time series?
#
# ### Other stuff to do:
#
# - undeerstand wtf sarima is and how to model seasonality
# - boxjenkins approach, see my notes
# - throw in other tests such as KPSS, ljung-box tests, portmanteau tests
# - residual analysis via qq plot and ACF, PACF to make sure I captured every pattern
# - the time series is multiplicative: need to transform it via log, as [suggested here](https://otexts.com/fpp2/components.html). The idea is that I am working with additive models, hence i should transform it to additive
# - make sure to calculate confidence intervals for predictions
# - model selection via AIC, BIC
# - I would like to make actual predictions also in the original scale
# - implement also holtwinters method
#
# After all of this, i shall move to transfer function modelling...
#

#

# ## Engineering Stuff

# ### STL Decomposition

# I am using 365 period for seasonality since stock prices have annual seasonality: similar periods across different years should see similar behavior (i.e. sell in may and go away, weakness in august-september, etc.)
#
# By looking at the plots, it seems that the variation increases with the level of the time series. This is probably a suggestion that it is multiplicative in nature, as stock price time series generally are, meaning that we could log-transform it to make it additive and easier to model.
#
# *NOTE*: [STL class reference](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/stl/_stl.pyx), cannot see docs because it is CPython

# +
stl = tsa_season.STL(
    smh_close_prices,
    period=365
)
res: tsa_season.DecomposeResult = stl.fit()

fig = res.plot()  # Using fig here avoids double plot
fig.set_size_inches(16, 12)
# -

# ### Log-Transform

smh_close_prices_log = smh_close_prices.apply(np.log)

# #### STL Decomposition

stl = tsa_season.STL(
    smh_close_prices_log, 
    period=365
)
res = stl.fit()
fig = res.plot()
fig.set_size_inches(16, 12)

# #### Classical Decomposition using MA - Additive

fig = tsa_season.seasonal_decompose(smh_close_prices_log, model='additive', period=365).plot()
fig.set_size_inches(16, 12)

# ### Differencing

smh_close_prices_log_diff = smh_close_prices_log.diff(periods=1).dropna()
fig = smh_close_prices_log_diff.plot(figsize=(16, 8))

smh_close_prices_log_diff.info

stl = tsa_season.STL(
    smh_close_prices_log_diff,
    period=365
)
res = stl.fit()
fig = res.plot()
fig.set_size_inches(16, 12)

res.resid.abs().sort_values(ascending=False)

fig = tsa_season.seasonal_decompose(smh_close_prices_log_diff, model='additive', period=365).plot()
fig.set_size_inches(16, 12)

# ## Stationarity Tests

# ### ADF Test

adf_res = tsa.adfuller(smh_close_prices)
adf_stat, adf_p_value = adf_res[0], adf_res[1]
print(adf_res)
print(adf_stat)
print(adf_p_value)

# ### KPSS Test

tsa.kpss(smh_close_prices)

#


