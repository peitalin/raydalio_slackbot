


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import datetime
import sys

# import pandas_datareader as pdr
# import lxml

# pdr.famafrench.get_available_datasets()
#  factornames = [
#     'F-F_Research_Data_Factors',
#     'F-F_Research_Data_Factors_weekly',
#     'F-F_Research_Data_Factors_daily' ]

# if PERIOD == 'd':
#     ffdat = pdr.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start=start, end=end)[0]
# else:
#     ffdat = pdr.DataReader('F-F_Research_Data_Factors', 'famafrench', start=start, end=end)[0]
# ffdat.rename(columns={"Mkt-RF": "MKT_PREM"}, inplace=True)
# SP500 = pdr.get_data_yahoo('^GSPC', start, end, interval=PERIOD).to_period(PERIOD)
# get stock price data for Icahn Enterprises "IEP"
# IEP = pdr.get_data_yahoo('IEP', start, end, interval=PERIOD).to_period(PERIOD)
# get stock price data for Berkshire Hathaway "BRK-A"
# BRKA = pdr.get_data_yahoo('BRK-A', start, end, interval=PERIOD).to_period(PERIOD)
# get stock price data for ClearBridge Value Truest "LMVTX"
# LMVTX = pdr.get_data_yahoo('LMVTX', start, end, interval=PERIOD).to_period(PERIOD)


def carl_icahn(start='2012-01-01', end='2017-01-01'):

    if PERIOD == 'd':
        IEP = pd.read_csv(raydir + "IEP_daily.csv")
    else:
        IEP = pd.read_csv(raydir + "IEP_monthly.csv")
    ##### Icahn Enterprises Excess Returns
    IEP.set_index("Date", inplace=1)
    IEP_returns = (np.log(IEP/IEP.shift()) * 100)
    IEP_returns.rename(columns={'Adj Close': 'IEP_returns'}, inplace=True)
    # join dataset with Fama-French Factors
    IEP_data1 = IEP_returns.join(SP500_excess_returns, how='inner')
    IEP_data3 = IEP_returns.join(ffdat, how='inner')[['IEP_returns', 'MKT_PREM', 'SMB', 'HML', 'RF']]
    # subtract risk-free from portfolio returns
    IEP_data1['IEP_excess_returns'] = IEP_data1['IEP_returns'] - IEP_data3['RF']
    IEP_data3['IEP_excess_returns'] = IEP_data3['IEP_returns'] - IEP_data3['RF']

    # estimate FF3-factor model (with start and end dates)
    model_1 = smf.ols("IEP_excess_returns ~ MKT_PREM", data=IEP_data1[start:end])
    model_3 = smf.ols("IEP_excess_returns ~ MKT_PREM + SMB + HML", data=IEP_data3[start:end])

    results1 = model_1.fit()
    results3 = model_3.fit()

    annualized_alpha1 = ((1 + results1.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval1 = results1.pvalues['Intercept']
    annualized_alpha3 = ((1 + results3.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval3 = results3.pvalues['Intercept']

    resStr1 = "\n\n\tPeriod: {} ~ {}".format(start, end)
    resStr1 += "\nAnnualized 1-factor alpha: {:.2f}%. ".format(annualized_alpha1)
    resStr1 += "\np-value: {:.4f}.".format(pval1)

    resStr3 = "\n\n\tPeriod: {} ~ {}".format(start, end)
    resStr3 += "\nAnnualized 3-factor alpha: {:.2f}%.".format(annualized_alpha3)
    resStr3 += "\np-value: {:.4f}.".format(pval3)

    summary1 = '\n'.join([
        '\n'.join(str(results1.summary()).split('\n')[:3]),
        '\n'.join(str(results1.summary()).split('\n')[7:8]),
        '\n'.join(str(results1.summary()).split('\n')[11:-8])
    ])
    summary3 = '\n'.join([
        '\n'.join(str(results3.summary()).split('\n')[:3]),
        '\n'.join(str(results3.summary()).split('\n')[7:8]),
        '\n'.join(str(results3.summary()).split('\n')[11:-8])
    ])

    return summary1 + '\n\n\n' + summary3 + resStr1 + resStr3



def warren_buffett(start='2012-01-01', end='2017-01-01'):

    if PERIOD == 'd':
        BRKA = pd.read_csv(raydir + "BRKA_daily.csv")
    else:
        BRKA = pd.read_csv(raydir + "BRKA_monthly.csv")
    ##### Berkshire Hathaway Excess Returns
    BRKA.set_index("Date", inplace=1)
    BRKA_returns = (np.log(BRKA/BRKA.shift()) * 100)
    BRKA_returns.rename(columns={'Adj Close': 'BRKA_returns'}, inplace=True)
    # join dataset with Fama-French Factors
    BRKA_data1 = BRKA_returns.join(SP500_excess_returns, how='inner')
    BRKA_data3 = BRKA_returns.join(ffdat, how='inner')[['BRKA_returns', 'MKT_PREM', 'SMB', 'HML', 'RF']]
    # subtract risk-free from portfolio returns
    BRKA_data1['BRKA_excess_returns'] = BRKA_data1['BRKA_returns'] - BRKA_data3['RF']
    BRKA_data3['BRKA_excess_returns'] = BRKA_data3['BRKA_returns'] - BRKA_data3['RF']

    # estimate FF3-factor model (with start and end dates)
    model_1 = smf.ols("BRKA_excess_returns ~ MKT_PREM", data=BRKA_data1[start:end])
    model_3 = smf.ols("BRKA_excess_returns ~ MKT_PREM + SMB + HML", data=BRKA_data3[start:end])

    results1 = model_1.fit()
    results3 = model_3.fit()

    annualized_alpha1 = ((1 + results1.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval1 = results1.pvalues['Intercept']
    annualized_alpha3 = ((1 + results3.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval3 = results3.pvalues['Intercept']

    resStr1 = "\n\n\tPeriod: {} ~ {}".format(start, end)
    resStr1 += "\nAnnualized 1-factor alpha: {:.2f}%. ".format(annualized_alpha1)
    resStr1 += "\np-value: {:.4f}.".format(pval1)

    resStr3 = "\n\n\tPeriod: {} ~ {}".format(start, end)
    resStr3 += "\nAnnualized 3-factor alpha: {:.2f}%.".format(annualized_alpha3)
    resStr3 += "\np-value: {:.4f}.".format(pval3)

    summary1 = '\n'.join([
        '\n'.join(str(results1.summary()).split('\n')[:3]),
        '\n'.join(str(results1.summary()).split('\n')[7:8]),
        '\n'.join(str(results1.summary()).split('\n')[11:-8])
    ])
    summary3 = '\n'.join([
        '\n'.join(str(results3.summary()).split('\n')[:3]),
        '\n'.join(str(results3.summary()).split('\n')[7:8]),
        '\n'.join(str(results3.summary()).split('\n')[11:-8])
    ])

    return summary1 + '\n\n\n' + summary3 + resStr1 + resStr3



def bill_miller(start='2012-01-01', end='2017-01-01'):

    if PERIOD == 'd':
        LMVTX = pd.read_csv(raydir + "LMVTX_daily.csv")
    else:
        LMVTX = pd.read_csv(raydir + "LMVTX_monthly.csv")
    ##### LMVTX: Clearbridge Value Trust
    LMVTX.set_index("Date", inplace=1)
    LMVTX_returns = (np.log(LMVTX/LMVTX.shift()) * 100)
    LMVTX_returns.rename(columns={'Adj Close': 'LMVTX_returns'}, inplace=True)
    # join dataset with Fama-French Factors
    LMVTX_data1 = LMVTX_returns.join(SP500_excess_returns, how='inner')
    LMVTX_data3 = LMVTX_returns.join(ffdat, how='inner')[['LMVTX_returns', 'MKT_PREM', 'SMB', 'HML', 'RF']]
    # subtract risk-free from portfolio returns
    LMVTX_data1['LMVTX_excess_returns'] = LMVTX_data1['LMVTX_returns'] - LMVTX_data3['RF']
    LMVTX_data3['LMVTX_excess_returns'] = LMVTX_data3['LMVTX_returns'] - LMVTX_data3['RF']

    # estimate FF3-factor model (with start and end dates)
    model_1 = smf.ols("LMVTX_excess_returns ~ MKT_PREM", data=LMVTX_data1[start:end])
    model_3 = smf.ols("LMVTX_excess_returns ~ MKT_PREM + SMB + HML", data=LMVTX_data3[start:end])

    results1 = model_1.fit()
    results3 = model_3.fit()

    annualized_alpha1 = ((1 + results1.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval1 = results1.pvalues['Intercept']
    annualized_alpha3 = ((1 + results3.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval3 = results3.pvalues['Intercept']

    resStr1 = "\n\n\tPeriod: {} ~ {}".format(start, end)
    resStr1 += "\nAnnualized 1-factor alpha: {:.2f}%. ".format(annualized_alpha1)
    resStr1 += "\np-value: {:.4f}.".format(pval1)

    resStr3 = "\n\n\tPeriod: {} ~ {}".format(start, end)
    resStr3 += "\nAnnualized 3-factor alpha: {:.2f}%.".format(annualized_alpha3)
    resStr3 += "\np-value: {:.4f}.".format(pval3)

    summary1 = '\n'.join([
        '\n'.join(str(results1.summary()).split('\n')[:3]),
        '\n'.join(str(results1.summary()).split('\n')[7:8]),
        '\n'.join(str(results1.summary()).split('\n')[11:-8])
    ])
    summary3 = '\n'.join([
        '\n'.join(str(results3.summary()).split('\n')[:3]),
        '\n'.join(str(results3.summary()).split('\n')[7:8]),
        '\n'.join(str(results3.summary()).split('\n')[11:-8])
    ])

    return summary1 + '\n\n\n' + summary3 + resStr1 + resStr3



def david_einhorn(start='2012-01-01', end='2016-12-01'):

    GLRE = pd.read_csv(raydir + "greenlight_capital.csv")
    ##### GLRE: Greenlight Capital RE
    GLRE.set_index("Date", inplace=1)
    # GLRE_returns = (np.log(GLRE/GLRE.shift()) * 100)
    GLRE_returns = GLRE
    GLRE_returns.rename(columns={'Greenlight Capital Re Ltd Class A': 'GLRE_returns'}, inplace=True)
    # join dataset with Fama-French Factors
    GLRE_data1 = GLRE_returns.join(SP500_excess_returns, how='inner')
    GLRE_data3 = GLRE_returns.join(ffdat, how='inner')[['GLRE_returns', 'MKT_PREM', 'SMB', 'HML', 'RF']]
    # subtract risk-free from portfolio returns
    GLRE_data1['GLRE_excess_returns'] = GLRE_data1['GLRE_returns'] - GLRE_data3['RF']
    GLRE_data3['GLRE_excess_returns'] = GLRE_data3['GLRE_returns'] - GLRE_data3['RF']

    # estimate FF3-factor model (with start and end dates)
    model_1 = smf.ols("GLRE_excess_returns ~ MKT_PREM", data=GLRE_data1[start:end])
    model_3 = smf.ols("GLRE_excess_returns ~ MKT_PREM + SMB + HML", data=GLRE_data3[start:end])

    results1 = model_1.fit()
    results3 = model_3.fit()

    annualized_alpha1 = ((1 + results1.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval1 = results1.pvalues['Intercept']
    annualized_alpha3 = ((1 + results3.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval3 = results3.pvalues['Intercept']

    resStr1 = "\n\n\tPeriod: {} ~ {}".format(start, end)
    resStr1 += "\nAnnualized 1-factor alpha: {:.2f}%. ".format(annualized_alpha1)
    resStr1 += "\np-value: {:.4f}.".format(pval1)

    resStr3 = "\n\n\tPeriod: {} ~ {}".format(start, end)
    resStr3 += "\nAnnualized 3-factor alpha: {:.2f}%.".format(annualized_alpha3)
    resStr3 += "\np-value: {:.4f}.".format(pval3)

    summary1 = '\n'.join([
        '\n'.join(str(results1.summary()).split('\n')[:3]),
        '\n'.join(str(results1.summary()).split('\n')[7:8]),
        '\n'.join(str(results1.summary()).split('\n')[11:-8])
    ])
    summary3 = '\n'.join([
        '\n'.join(str(results3.summary()).split('\n')[:3]),
        '\n'.join(str(results3.summary()).split('\n')[7:8]),
        '\n'.join(str(results3.summary()).split('\n')[11:-8])
    ])

    return summary1 + '\n\n\n' + summary3 + resStr1 + resStr3



def all_weather(start='2012-01-01', end='2016-12-01'):

    ALL_WEATHER = pd.read_csv(raydir + "bridgewater_all_weather.csv")
    ##### Bridgewater All-Weather Portfolio
    ALL_WEATHER.set_index("Date", inplace=1)
    # ALL_WEATHER_returns = (np.log(ALL_WEATHER/ALL_WEATHER.shift()) * 100)
    ALL_WEATHER_returns = ALL_WEATHER
    ALL_WEATHER_returns.rename(columns={'Bridgewater All Weather 12% Strategy': 'ALL_WEATHER_returns'}, inplace=True)
    # join dataset with Fama-French Factors
    ALL_WEATHER_data1 = ALL_WEATHER_returns.join(SP500_excess_returns, how='inner')
    ALL_WEATHER_data3 = ALL_WEATHER_returns.join(ffdat, how='inner')[['ALL_WEATHER_returns', 'MKT_PREM', 'SMB', 'HML', 'RF']]
    # subtract risk-free from portfolio returns
    ALL_WEATHER_data1['ALL_WEATHER_excess_returns'] = ALL_WEATHER_data1['ALL_WEATHER_returns'] - ALL_WEATHER_data3['RF']
    ALL_WEATHER_data3['ALL_WEATHER_excess_returns'] = ALL_WEATHER_data3['ALL_WEATHER_returns'] - ALL_WEATHER_data3['RF']

    # estimate FF3-factor model (with start and end dates)
    model_1 = smf.ols("ALL_WEATHER_excess_returns ~ MKT_PREM", data=ALL_WEATHER_data1[start:end])
    model_3 = smf.ols("ALL_WEATHER_excess_returns ~ MKT_PREM + SMB + HML", data=ALL_WEATHER_data3[start:end])

    results1 = model_1.fit()
    results3 = model_3.fit()

    annualized_alpha1 = ((1 + results1.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval1 = results1.pvalues['Intercept']
    annualized_alpha3 = ((1 + results3.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval3 = results3.pvalues['Intercept']

    resStr1 = "\n\n\tPeriod: {} ~ {}".format(start, end)
    resStr1 += "\nAnnualized 1-factor alpha: {:.2f}%. ".format(annualized_alpha1)
    resStr1 += "\np-value: {:.4f}.".format(pval1)

    resStr3 = "\n\n\tPeriod: {} ~ {}".format(start, end)
    resStr3 += "\nAnnualized 3-factor alpha: {:.2f}%.".format(annualized_alpha3)
    resStr3 += "\np-value: {:.4f}.".format(pval3)

    summary1 = '\n'.join([
        '\n'.join(str(results1.summary()).split('\n')[:3]),
        '\n'.join(str(results1.summary()).split('\n')[7:8]),
        '\n'.join(str(results1.summary()).split('\n')[11:-8])
    ])
    summary3 = '\n'.join([
        '\n'.join(str(results3.summary()).split('\n')[:3]),
        '\n'.join(str(results3.summary()).split('\n')[7:8]),
        '\n'.join(str(results3.summary()).split('\n')[11:-8])
    ])

    return summary1 + '\n\n\n' + summary3 + resStr1 + resStr3


def pure_alpha(start='2012-01-01', end='2016-12-01'):

    PURE_ALPHA = pd.read_csv(raydir + "bridgewater_pure_alpha.csv")
    ##### Bridgewater Pure-Alpha 18% Volatility fund
    PURE_ALPHA.set_index("Date", inplace=1)
    # PURE_ALPHA_returns = (np.log(PURE_ALPHA/PURE_ALPHA.shift()) * 100)
    PURE_ALPHA_returns = PURE_ALPHA
    PURE_ALPHA_returns.rename(columns={'Bridgewater Pure Alpha Strat 18% Vol': 'PURE_ALPHA_returns'}, inplace=True)
    # join dataset with Fama-French Factors
    PURE_ALPHA_data1 = PURE_ALPHA_returns.join(SP500_excess_returns, how='inner')
    PURE_ALPHA_data3 = PURE_ALPHA_returns.join(ffdat, how='inner')[['PURE_ALPHA_returns', 'MKT_PREM', 'SMB', 'HML', 'RF']]
    # subtract risk-free from portfolio returns
    PURE_ALPHA_data1['PURE_ALPHA_excess_returns'] = PURE_ALPHA_data1['PURE_ALPHA_returns'] - PURE_ALPHA_data3['RF']
    PURE_ALPHA_data3['PURE_ALPHA_excess_returns'] = PURE_ALPHA_data3['PURE_ALPHA_returns'] - PURE_ALPHA_data3['RF']

    # estimate FF3-factor model (with start and end dates)
    model_1 = smf.ols("PURE_ALPHA_excess_returns ~ MKT_PREM", data=PURE_ALPHA_data1[start:end])
    model_3 = smf.ols("PURE_ALPHA_excess_returns ~ MKT_PREM + SMB + HML", data=PURE_ALPHA_data3[start:end])

    results1 = model_1.fit()
    results3 = model_3.fit()

    annualized_alpha1 = ((1 + results1.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval1 = results1.pvalues['Intercept']
    annualized_alpha3 = ((1 + results3.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval3 = results3.pvalues['Intercept']

    resStr1 = "\n\n\tPeriod: {} ~ {}".format(start, end)
    resStr1 += "\nAnnualized 1-factor alpha: {:.2f}%. ".format(annualized_alpha1)
    resStr1 += "\np-value: {:.4f}.".format(pval1)

    resStr3 = "\n\n\tPeriod: {} ~ {}".format(start, end)
    resStr3 += "\nAnnualized 3-factor alpha: {:.2f}%.".format(annualized_alpha3)
    resStr3 += "\np-value: {:.4f}.".format(pval3)

    summary1 = '\n'.join([
        '\n'.join(str(results1.summary()).split('\n')[:3]),
        '\n'.join(str(results1.summary()).split('\n')[7:8]),
        '\n'.join(str(results1.summary()).split('\n')[11:-8])
    ])
    summary3 = '\n'.join([
        '\n'.join(str(results3.summary()).split('\n')[:3]),
        '\n'.join(str(results3.summary()).split('\n')[7:8]),
        '\n'.join(str(results3.summary()).split('\n')[11:-8])
    ])

    return summary1 + '\n\n\n' + summary3 + resStr1 + resStr3



###############################################################
########## Setup Risk Factor Data

PERIOD = 'm'
PERIODS_PER_ANNUM = { 'm': 12, 'd': 252 }
start = datetime.datetime(1980, 1, 1)
end = datetime.datetime(2017, 1, 30)

raydir = "./raydalio_slackbot/"
# raydir = "./"

if PERIOD=='d':
    ffdat = pd.read_csv(raydir + "ffdat_daily.csv")
    SP500 = pd.read_csv(raydir + "SP500_daily.csv")
else:
    ffdat = pd.read_csv(raydir + "ffdat_monthly.csv")
    SP500 = pd.read_csv(raydir + "SP500_monthly.csv")

### FAMA-FRENCH Factors
ffdat.set_index("Date", inplace=True)
#### S&P500
SP500.set_index("Date", inplace=True)
SP500_returns = (np.log(SP500/SP500.shift()) * 100)
SP500_excess_returns = SP500_returns['Adj Close'] - ffdat['RF']
SP500_excess_returns.name = 'MKT_PREM'



if __name__=="__main__":

    # print(sys.argv[4])
    # if 'm' in sys.argv[4]:
    #     PERIOD = 'm'
    # else:
    #     PERIOD = 'd'

    investor = sys.argv[1].lower()
    if "carl" in investor or "icahn" in investor:
        print(carl_icahn(sys.argv[2], sys.argv[3]))

    if "warren" in investor or "buffet" in investor:
        print(warren_buffett(sys.argv[2], sys.argv[3]))

    if "bill" in investor or "miller" in investor:
        print(warren_buffett(sys.argv[2], sys.argv[3]))

    if "david" in investor or "einhorn" in investor:
        print(david_einhorn(sys.argv[2], sys.argv[3]))

    if "all-weather" in investor:
        print(all_weather(sys.argv[2], sys.argv[3]))

    if "pure-alpha" in investor:
        print(pure_alpha(sys.argv[2], sys.argv[3]))

    if "ray" in investor or "dalio" in investor:
        print("Try 'ray dalio all weather' or 'ray dalio pure alpha' for 'investor' parameter")





