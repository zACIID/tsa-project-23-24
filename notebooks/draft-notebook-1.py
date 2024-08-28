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
# -



# ## EDA

# index_col=0 so that timestamp is the index -> handy for plots
smh = pd.read_csv('../data/smh-2000-06-05_2024-08-23.csv', parse_dates=True, index_col=0)
smh.describe()

smh.head(10)

smh.plot(subplots=True, figsize=(15, 15))

# We are only interested in closing prices



# Just take the Series (i.e. the column) so that it is easier to handle
smh_close_prices = smh["Close"].sort_index(ascending=True)

smh_close_prices.plot(figsize=(12, 8))

#

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

# ### Classical Decomposition using MA - Additive

fig = tsa_season.seasonal_decompose(smh_close_prices, model='additive', period=365).plot()
fig.set_size_inches(16, 12)

# ### Classical Decomposition using MA - Multiplicative

fig = tsa_season.seasonal_decompose(smh_close_prices, model='multiplicative', period=365).plot()
fig.set_size_inches(16, 12)

# ### Autocorrelation Plots
#
# Note: [How many lags to show in (P)ACF plot](https://stats.stackexchange.com/a/81432/395870)

fig = tsa_plt.plot_acf(smh_close_prices, lags=100, auto_ylims=True)
fig.set_size_inches(12, 8)

fig = tsa_plt.plot_acf(smh_close_prices, lags=40, auto_ylims=True)
fig.set_size_inches(12, 8)

fig = tsa_plt.plot_pacf(smh_close_prices, lags=40, auto_ylims=True)
fig.set_size_inches(12, 8)

# ### Stationarity Tests

# #### ADF Test

adf_res = tsa.adfuller(smh_close_prices)
adf_stat, adf_p_value = adf_res[0], adf_res[1]
print(adf_res)
print(adf_stat)
print(adf_p_value)

# #### KPSS Test

tsa.kpss(smh_close_prices)

# ## Differencing

smh_close_prices_diff = smh_close_prices.diff(periods=1).dropna()
fig = smh_close_prices_diff.plot(figsize=(16, 8))

smh_close_prices_diff.info

stl = tsa_season.STL(
    smh_close_prices_diff,
    period=365
)
res = stl.fit()
fig = res.plot()
fig.set_size_inches(16, 12)

fig = tsa_season.seasonal_decompose(smh_close_prices_diff, model='additive', period=365).plot()
fig.set_size_inches(16, 12)

# ### Biggest Residuals
#
# May be helpful to identify outliers

res.resid.abs().sort_values(ascending=False)

# ### Autocorrelation Plots
#
# Note: [How many lags to show in (P)ACF plot](https://stats.stackexchange.com/a/81432/395870)

fig = tsa_plt.plot_acf(smh_close_prices_diff, lags=250, auto_ylims=True)
fig.set_size_inches(12, 8)

# Without the assignment it shows 2 plots for some reason...
fig = tsa_plt.plot_acf(smh_close_prices_diff, lags=40, auto_ylims=True)
fig.set_size_inches(12, 8)



# #### Lagplots for lags with strongest autocorr
#
# Taking the top 10 (lag 0 excluded)
# Maybe comment on these... the price from 1, 8 and 16 days prior seems to be the most significant? i.e. basically the same day from the prior week/two-weeks?

# +
# Extract the top k autocorr scores
acf_res, confints = tsa.acf(smh_close_prices_diff, nlags=40, alpha=0.05)
confints = confints.T  # want to have shape (2, nlags) instead of (nlags, 2) so calculating the mask is easier

top_k = 10
top_autocorr = pd.DataFrame({
    # top_k + 1 because lag 0 is not really interesting
    "lag": np.argsort(-np.abs(acf_res))[:top_k+1],
    "autocorr": np.sort(-np.abs(acf_res))[:top_k+1]
})
top_autocorr

# +
# Plot every lag except 0
to_plot = top_autocorr["lag"]
nrows, ncols = len(to_plot) // 2 + 1, 2
fig, axs = plt.subplots(nrows, ncols, figsize=(20, nrows*3.5))

for i, lag in enumerate(to_plot[1:]):
    pd_plt.lag_plot(smh_close_prices_diff, ax=axs[i // ncols, i % ncols], lag=lag)

# +
fig, ax = plt.subplots(figsize=(12, 8))

# Without the assignment it shows 2 plots for some reason...
fig = tsa_plt.plot_pacf(smh_close_prices_diff, ax=ax, auto_ylims=True)
# -

# ### Stationarity Tests

# #### ADF Test

adf_res = tsa.adfuller(smh_close_prices_diff)
adf_stat, adf_p_value = adf_res[0], adf_res[1]
print(adf_res)
print(adf_stat)
print(adf_p_value)

# #### KPSS Test

tsa.kpss(smh_close_prices_diff)

#

# ## Log-Transform

smh_close_prices_log = smh_close_prices.apply(np.log)

# #### STL Decomposition

stl = tsa_season.STL(
    smh_close_prices_log, 
    period=365
)
res = stl.fit()
fig = res.plot()
fig.set_size_inches(16, 12)

smh_close_prices[smh_close_prices_log.index.year == 2001].info



x = smh_close_prices_log.copy()
x.index = smh_close_prices.index
stl = tsa_season.STL(
    x,
    period=248,
    seasonal=249
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

x = tsa_season.seasonal_decompose(smh_close_prices_log_diff, model='additive', period=365)

fig = tsa_season.seasonal_decompose(smh_close_prices_log_diff, model='additive', period=365).plot()
fig.set_size_inches(16, 12)

# ### Biggest Residuals
#
# May be helpful to identify outliers

res.resid.abs().sort_values(ascending=False)

# ### Autocorrelation Plots
#
# Note: [How many lags to show in (P)ACF plot](https://stats.stackexchange.com/a/81432/395870)

fig = tsa_plt.plot_acf(smh_close_prices_log_diff, lags=250, auto_ylims=True)
fig.set_size_inches(12, 8)

# Without the assignment it shows 2 plots for some reason...
fig = tsa_plt.plot_acf(smh_close_prices_log_diff, lags=40, auto_ylims=True)
fig.set_size_inches(12, 8)

acf_res = tsa.acf(smh_close_prices_log_diff, nlags=40)

# #### Lagplots for lags with strongest autocorr
#
# Taking the top 10 (lag 0 excluded)
# Maybe comment on these... the price from 1, 8 and 16 days prior seems to be the most significant? i.e. basically the same day from the prior week/two-weeks?

print(np.argsort(-np.abs(acf_res))[1:11])
print(np.sort(-np.abs(acf_res))[1:11])
fig, axs = plt.subplots(5, 2, figsize=(15, 25))
for i, lag in enumerate(np.argsort(-np.abs(acf_res))[1:11]):
    pd_plt.lag_plot(smh_close_prices_log_diff, ax=axs[i % 5, i % 2], lag=lag)

# +
fig, ax = plt.subplots(figsize=(12, 8))

# Without the assignment it shows 2 plots for some reason...
fig = tsa_plt.plot_pacf(smh_close_prices_log_diff, ax=ax, auto_ylims=True)
# -

# ### Stationarity Tests



# #### ADF Test

adf_res = tsa.adfuller(smh_close_prices_log_diff)
adf_stat, adf_p_value = adf_res[0], adf_res[1]
print(adf_res)
print(adf_stat)
print(adf_p_value)

# #### KPSS Test

tsa.kpss(smh_close_prices_log_diff)

#



# ## SARIMA Fitting (on log series)

# ### Train Test Split

# Convert to period so that pmdarima can better leverage timestamp information
# Cannot do it at the beginning during EDA, else plots break
smh_close_prices_log.index = smh_close_prices_log.index.to_period('D')
smh_close_prices_log_diff.index = smh_close_prices_log_diff.index.to_period('D')

#

# +
train, test = pm_modsel.train_test_split(smh_close_prices_log, train_size=0.8)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast

# + active=""
#
#
# stlf = STLForecast(train, ARIMA, period=248, seasonal=249, model_kwargs=dict(order=(3, 1, 3)))
# stlf_res = stlf.fit()
#
# forecast = stlf_res.forecast(steps=test.shape[0])
#
# x = train.copy()
# x.index = x.index.to_timestamp()
# y = forecast
# y.index = test.index.to_timestamp()
# plt.plot(x)
# plt.plot(y)
# plt.show()
# -



# Fit a simple auto_arima model
with pm.StepwiseContext(max_steps=25):
    stl: tsa_season.DecomposeResult = tsa_season.STL(train, period=248, seasonal=249).fit()
    deseasonaled = stl.trend + stl.resid
    modl: pm.ARIMA = pm.auto_arima(deseasonaled,
                     start_p=1, d=1, start_q=0, max_p=10, max_d=1, max_q=10,
                     start_P=1, D=1, start_Q=0, max_P=5, max_D=1, max_Q=5, m=8, seasonal=True, 
                         max_order=8,
                     stationary=False,
                     information_criterion='aic',
                     suppress_warnings=False, error_action='trace', trace=True,
                         #method='powell',
                        stepwise=True, random=False, random_state=42, n_jobs=4,
                         sarimax_kwargs={
                             #"enforce_stationarity": False, "enforce_invertibility": False,
                         }
                     )

modl.summary()

res = modl.arima_res_
fig = res.plot_diagnostics()
fig.set_size_inches(20, 16)
preds = res.get_prediction()

# + active=""
# model = SARIMAX(train,
#                 order=(1, 1, 1), # You may need to adjust these parameters
#                 seasonal_order=(1, 1, 1, 7), # Adjust the seasonal_order as needed
#                 enforce_stationarity=True,
#                 enforce_invertibility=True).fit()

# + active=""
# forecast = model.get_forecast(steps=len(test))

# +
# Create predictions for the future, evaluate on test
#modl = pm.ARIMA(order=(3, 1, 1), seasonal_order=(1, 0, 1, 7)).fit(train)
stlf = STLForecast(train, ARIMA, period=248, seasonal=249, model_kwargs={
    "order": modl.order,
    "seasonal_order": modl.seasonal_order
})
stlf_res = stlf.fit()

# forecast = stlf_res.forecast(steps=test.shape[0])
forecast = stlf_res.get_prediction(start=train.shape[0], end=train.shape[0] + test.shape[0])

preds, conf_int = modl.predict(n_periods=test.shape[0], return_conf_int=True)
in_sample_preds, conf_int_in_sample = modl.predict_in_sample(start=21, end=train.shape[0], return_conf_int=True)

# Print the error:
print("Test RMSE: %.3f" % np.sqrt(sk_metrics.root_mean_squared_error(test, preds)))
print("Test MAPE: %.3f" % np.sqrt(sk_metrics.mean_absolute_percentage_error(test, preds)))

# +
# #############################################################################
# Plot the points and the forecasts
x_axis = np.arange(train.shape[0] + preds.shape[0])
x_years = x_axis + 2000

plt.figure(figsize=(12, 8))
plt.plot(x_years[x_axis[:train.shape[0]]], train, alpha=0.75)
plt.scatter(x_years[x_axis[train.shape[0]:]], test,
            alpha=0.4, marker='x')  # Test data

plt.plot(x_years[x_axis[20:train.shape[0]]], in_sample_preds, alpha=0.75)
plt.fill_between(x_years[x_axis[20:in_sample_preds.shape[0]]], conf_int_in_sample[20:, 0], conf_int_in_sample[20:, 1], alpha=0.1, color='green')
#plt.plot(x_years[x_axis[train.shape[0]:]], preds, alpha=0.75)  # Forecasts
#plt.fill_between(x_years[x_axis[-preds.shape[0]:]], conf_int[:, 0], conf_int[:, 1], alpha=0.1, color='b')
plt.plot(x_years[x_axis[train.shape[0]-1:]], forecast.predicted_mean, alpha=0.75)  # Forecasts
plt.fill_between(x_years[x_axis[-(test.shape[0]+1):]], forecast.conf_int().iloc[:, 0], forecast.conf_int().iloc[:, 1], alpha=0.1, color='b')

plt.title("Lynx forecasts")
plt.xlabel("Year")

# -

# ### Auto ARIMA



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
# ### Ideas: stuff to do
#
# - put my whole suite of EDA/stationarity analysis tools into one method that I will call for the following time series:
#     - vanilla
#     - differenced
#     - log
#     - log+differenced 
# - train-split before everything else, esle I am drawing conclusions for the model on test data too... ACF stuff on train only because it influences my decision on model params
# - consider seasonality of 7 or 8 because possible evidence in the correlogram plot
#     - try to fit models with autoarima, but also by hand with seasonality 1,7,8 and AR/MA order < seasonality and dictated by the ACF/PACF plot (need to understand why the PACF plot tells me stuff about MA process)
#     - need to comment on this properly though... because seasonality looks weak (how to understand if weak seasonal component? just look at magnitude of STL?) in the non-diffed log-series when compared to trend (in the diffed series it looks better because 1-diff removes poly trend of degree 1)
#     - also professor himself said that seasonal component is weak in stocks... I would need to find real evidence to say that seasonality is annual, 365...
# - if poor results, may want to try on a set that looks clean (i.e. without 2020 crash, or 2008 crisis) just to see what happens... do not know if this is actually justifiable becuase that is important variance that we are discarding
#     - but maybe this I should try to see if SARIMA does well -> conclusion may be "if no extreme events SARIMA would seem to do good..."
#     - if ARIMA prediction is poor but true thing is still inside the confidence band and trend seems right, then I think it is a partial win (need to understand how to interpret confidence band, when they are good or too imprecise...)
# - make sure to do residual analysis...
# - **after all this, try ETS...** 
# - If the consideration right before the "Next steps" section of [this article](https://www.quantstart.com/articles/Autoregressive-Integrated-Moving-Average-ARIMA-p-d-q-Models-for-Time-Series-Analysis/) is right, then I may want to try and fit a SETAR model? i.e. something non-linear that could be able to deal with volatilty clustering/periods of different volatilty
#
# ### Explanations on why models aren't good
#
# - SARIMA: 
#     - quote from https://www.quantstart.com/articles/Autoregressive-Moving-Average-ARMA-p-q-Models-for-Time-Series-Analysis-Part-3/
#     > Note that an ARMA model does not take into account volatility clustering, a key empirical phenomena of many financial time series. It is not a conditionally heteroscedastic model. For that we will need to wait for the ARCH and GARCH models.
#     - todo maybe there just isn't anough info to fit an ARIMA model... acf plots look like shit because spikes are very close to the white noise bands
#
#
# ### Motivation of other stuff
#
# - want to log-transform because economic time series are usually multiplicative in nature, and additive models are somewhat easier to handle. Additionally, KPSS stat requires a linear-trend time series because it tries to verify trend-stationarity
# - I am trimming the first part of the smhprice time series because it isn't a full year, so I am actually starting from 2001. My idea is that it is better to have full periods so that there is an equal representation of seasonality in the time series, also because I have a suspect that 
#

# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_seasonal_boxplots(data: pd.Series, seasonality):
    """
    Slices the time series into K seasons of period P and plots the distribution of each season via boxplots.

    Parameters:
    data (pd.Series): The time series data with a PeriodIndex.
    P (str): The period string to define a season (e.g., 'M' for months, 'Q' for quarters).
    K (int): The number of seasons to slice.

    Returns:
    None: Displays the boxplot.
    """
    
    # Resample 
    seasons_data = {}
    data = data.copy()[(2000 < data.index.year) & (data.index.year < 2024)]
    # for i in range(1, seasonality+1):
    #     season_label = f'Period {i+1}'
    #     # season_data = data.resample(f"{i}B", offset=f"{i}B")
    #     season_data = data.resample(f"{seasonality}M", offset=pd.Timedelta(i)).first()
    #     seasons_data[season_label] = season_data.values
    for month in range(1, 13):
        month_label = f'Month {month}'
        seasons_data[month_label] = data[data.index.month == month].values

    # Plot the boxplot for each season
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=seasons_data)

    # Adding title and labels
    #plt.title(f'Boxplots of {K} Seasons of Period {P}')
    plt.xlabel('Season')
    plt.ylabel('Value')

    # Show the plot
    plt.show()

# Example usage:
# Assuming `data` is your pandas Series with a PeriodIndex
plot_seasonal_boxplots(smh_close_prices, seasonality=12)

# -


