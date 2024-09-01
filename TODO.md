# TODOs

## Stuff to do in the project

- Preliminary: I should transform the .xlsx of the semiconductor sales TS into a .csv prior to the analysis, so that it is readily available
- **First task: stock price prediction**
    - *First section: data exploration*
        - plot of the smh time series to check them and compare them
        - ACF, PACF plots
            - 95% confidence bands are valid if we have "a large enough number of points". Is my sample size enough?
            - comment the plots to describe how the series seems to behave
                - see interpretation of ACF plot at page 24 of tsa notes
        - lagplots to understand if the relationship between a point and its lagged version is actually linear, in which case autocorrelation makes sense - else it doesn't, should probably look at Average Mutual Information
        - Decompositions:
            - Classic and/or STL
            - comment on the components (behavior, impact in terms of absolute values, ...)
        - other stuff? #todo
    - *Engineering*
        - Filtering/Smoothing? #todo when should I use filtering/smoothing?
        - Try differencing if #todo (if what? there is still autocorrelation left in the noise? I don't remember atm)
            - I actually think I should use differencing if the series needs detrending because it is not stationary. How to check for stationarity? various tests (ADF, KPSS) and possibly an analysis of rolling mean and std: they should be independent of time windows (covariance only dependent on lag) and constant for a given lag
    - *Modelling*
        - Holt-Winters with seasons
        - SARIMA
            - check the open articles on SARIMA and stock prediction
        - which non linear model to try? SETAR? LSTM??? #todo
            - especially if non linear relationships are highlighted by lagplots
- **Second task: Transfer Function Modelling**
    - *First section: data exploration*:
        - Create a time series from the SMH price that is in the same scale as that of the semiconductor sales
        - *same framework as the data expl. section for the previous task*
    - *Second section: engineering*
        - prewhitening
        - #todo ????
    - *Third section: modelling*
        - transfer function modelling how?

### Modelling Procedure (from TSA notes)

Modelling procedure
1. Plot the data and identify unusual observations that could potentially
   affect the result.
2. If necessary use box-cox transformations to reduce the variance.
3. If not stationary, take first differencing (moving from ARIMA to ARMA)
   to make the process stationary.
4. Examine ACF and PACF to understand AR(p) and MA(q) components.
5. Fit chosen models and use IC to select the best one.
6. Check residuals using ACF or Portmanteau tests.
7. If residuals behave like white noise, calculate forecasts.




## TODO stuff to do next:
- i need to make the series stationary so that I can use SARIMA models. Why is it that I need stationarity for SARIMA models? I do not remember actually...
- I need to difference the time series until i get to stationarity, i.e. I remove Autocorrelation... careful of overdifferencing. Why is it that autocorr is 0 for stationary time series?

### Other stuff to do:

- undeerstand wtf sarima is and how to model seasonality
- boxjenkins approach, see my notes
- throw in other tests such as KPSS, ljung-box tests, portmanteau tests
- residual analysis via qq plot and ACF, PACF to make sure I captured every pattern
- the time series is multiplicative: need to transform it via log, as [suggested here](https://otexts.com/fpp2/components.html). The idea is that I am working with additive models, hence i should transform it to additive
- make sure to calculate confidence intervals for predictions
- model selection via AIC, BIC
- I would like to make actual predictions also in the original scale
- implement also holtwinters method

After all of this, i shall move to transfer function modelling...


### Ideas: stuff to do

- put my whole suite of EDA/stationarity analysis tools into one method that I will call for the following time series:
    - vanilla
    - differenced
    - log
    - log+differenced
- train-split before everything else, esle I am drawing conclusions for the model on test data too... ACF stuff on train only because it influences my decision on model params
- consider seasonality of 7 or 8 because possible evidence in the correlogram plot
    - try to fit models with autoarima, but also by hand with seasonality 1,7,8 and AR/MA order < seasonality and dictated by the ACF/PACF plot (need to understand why the PACF plot tells me stuff about MA process)
    - need to comment on this properly though... because seasonality looks weak (how to understand if weak seasonal component? just look at magnitude of STL?) in the non-diffed log-series when compared to trend (in the diffed series it looks better because 1-diff removes poly trend of degree 1)
    - also professor himself said that seasonal component is weak in stocks... I would need to find real evidence to say that seasonality is annual, 365...
- if poor results, may want to try on a set that looks clean (i.e. without 2020 crash, or 2008 crisis) just to see what happens... do not know if this is actually justifiable becuase that is important variance that we are discarding
    - but maybe this I should try to see if SARIMA does well -> conclusion may be "if no extreme events SARIMA would seem to do good..."
    - if ARIMA prediction is poor but true thing is still inside the confidence band and trend seems right, then I think it is a partial win (need to understand how to interpret confidence band, when they are good or too imprecise...)
- make sure to do residual analysis...
- **after all this, try ETS...**
- If the consideration right before the "Next steps" section of [this article](https://www.quantstart.com/articles/Autoregressive-Integrated-Moving-Average-ARIMA-p-d-q-Models-for-Time-Series-Analysis/) is right, then I may want to try and fit a SETAR model? i.e. something non-linear that could be able to deal with volatilty clustering/periods of different volatilty

### Explanations on why models aren't good

- SARIMA:
    - quote from https://www.quantstart.com/articles/Autoregressive-Moving-Average-ARMA-p-q-Models-for-Time-Series-Analysis-Part-3/
  > Note that an ARMA model does not take into account volatility clustering, a key empirical phenomena of many financial time series. It is not a conditionally heteroscedastic model. For that we will need to wait for the ARCH and GARCH models.
    - todo maybe there just isn't anough info to fit an ARIMA model... acf plots look like shit because spikes are very close to the white noise bands


### Motivation of other stuff

- want to log-transform because economic time series are usually multiplicative in nature, and additive models are somewhat easier to handle. Additionally, KPSS stat requires a linear-trend time series because it tries to verify trend-stationarity
- I am trimming the first part of the smhprice time series because it isn't a full year, so I am actually starting from 2001. My idea is that it is better to have full periods so that there is an equal representation of seasonality in the time series, also because I have a suspect that 
- boxplots about annual seasonality of SMH may indicate that volatility does vary across months, but no clear seasonal pattern, since the median of every month seems to be the same 
