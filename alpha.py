


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


def carl_icahn(start='2012-01-01', end='2017-01-01'):

    # join dataset with Fama-French Factors
    data1 = IEP_returns.join(SP500_excess_returns, how='inner')
    data3 = IEP_returns.join(ffdat, how='inner')[['IEP_returns', 'MKT_PREM', 'SMB', 'HML', 'RF']]
    # subtract risk-free from portfolio returns
    data1['IEP_excess_returns'] = data1['IEP_returns'] - data3['RF']
    data3['IEP_excess_returns'] = data3['IEP_returns'] - data3['RF']

    # estimate FF3-factor model (with start and end dates)
    model_1 = smf.ols("IEP_excess_returns ~ MKT_PREM", data=data1[start:end])
    model_3 = smf.ols("IEP_excess_returns ~ MKT_PREM + SMB + HML", data=data3[start:end])

    results1 = model_1.fit()
    results3 = model_3.fit()

    annualized_alpha1 = ((1 + results1.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval1 = results1.pvalues['Intercept']
    annualized_alpha3 = ((1 + results3.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval3 = results3.pvalues['Intercept']

    print(results1.summary())
    print(results3.summary())
    print("\n\n\tPeriod: {} ~ {}".format(start, end))

    print("\nAnnualized 1-factor alpha of: {:.2f}%.".format(annualized_alpha1))
    if (pval1 > 0.05):
        print("Not significant at the 5% level: p-value: {:.4f}".format(pval1))
    else:
        print("Significant at the 5% level: p-value: {:.4f}".format(pval1))

    print("\nAnnualized 3-factor alpha of: {:.2f}%.".format(annualized_alpha3))
    if (pval3 > 0.05):
        print("Not significant at the 5% level: p-value: {:.4f}".format(pval3))
    else:
        print("Significant at the 5% level: p-value: {:.4f}".format(pval3))

    return results1.summary() + '\n' + results3.summary()


def warren_buffett(start='2012-01-01', end='2017-01-01'):

    # join dataset with Fama-French Factors
    data1 = BRKA_returns.join(SP500_excess_returns, how='inner')
    data3 = BRKA_returns.join(ffdat, how='inner')[['BRKA_returns', 'MKT_PREM', 'SMB', 'HML', 'RF']]
    # subtract risk-free from portfolio returns
    data1['BRKA_excess_returns'] = data1['BRKA_returns'] - data3['RF']
    data3['BRKA_excess_returns'] = data3['BRKA_returns'] - data3['RF']

    # estimate FF3-factor model (with start and end dates)
    model_1 = smf.ols("BRKA_excess_returns ~ MKT_PREM", data=data1[start:end])
    model_3 = smf.ols("BRKA_excess_returns ~ MKT_PREM + SMB + HML", data=data3[start:end])

    results1 = model_1.fit()
    results3 = model_3.fit()

    annualized_alpha1 = ((1 + results1.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval1 = results1.pvalues['Intercept']
    annualized_alpha3 = ((1 + results3.params['Intercept']/100) ** PERIODS_PER_ANNUM[PERIOD] - 1) * 100
    pval3 = results3.pvalues['Intercept']

    print(results1.summary())
    print(results3.summary())
    print("\n\n\tPeriod: {} ~ {}".format(start, end))

    print("\nAnnualized 1-factor alpha of: {:.2f}%.".format(annualized_alpha1))
    if (pval1 > 0.05):
        print("Not significant at the 5% level: p-value: {:.4f}".format(pval1))
    else:
        print("Significant at the 5% level: p-value: {:.4f}".format(pval1))

    print("\nAnnualized 3-factor alpha of: {:.2f}%.".format(annualized_alpha3))
    if (pval3 > 0.05):
        print("Not significant at the 5% level: p-value: {:.4f}".format(pval3))
    else:
        print("Significant at the 5% level: p-value: {:.4f}".format(pval3))

    return results1.summary() + '\n' + results3.summary()



if __name__=="__main__":

    print(sys.argv[4])
    if 'm' in sys.argv[4]:
        PERIOD = 'm'
    else:
        PERIOD = 'd'

    PERIODS_PER_ANNUM = { 'm': 12, 'd': 252 }
    start = datetime.datetime(1980, 1, 1)
    end = datetime.datetime(2017, 1, 30)


    if PERIOD == 'd':
        ffdat = pd.read_csv("ffdat_daily.csv")
        SP500 = pd.read_csv("SP500_daily.csv")
        IEP = pd.read_csv("IEP_daily.csv")
        BRKA = pd.read_csv("BRKA_daily.csv")
    else:
        ffdat = pd.read_csv("ffdat_monthly.csv")
        SP500 = pd.read_csv("SP500_monthly.csv")
        IEP = pd.read_csv("IEP_monthly.csv")
        BRKA = pd.read_csv("BRKA_monthly.csv")


    ### FAMA-FRENCH Factors
    ffdat.set_index("Date", inplace=True)

    #### S&P500
    SP500.set_index("Date", inplace=True)
    SP500_returns = (np.log(SP500/SP500.shift()) * 100)
    SP500_excess_returns = SP500_returns['Adj Close'] - ffdat['RF']
    SP500_excess_returns.name = 'MKT_PREM'


    ##### Icahn Enterprises
    IEP.set_index("Date", inplace=1)
    IEP_returns = (np.log(IEP/IEP.shift()) * 100)
    IEP_returns.rename(columns={'Adj Close': 'IEP_returns'}, inplace=True)

    ##### Berkshire Hathaway
    BRKA.set_index("Date", inplace=1)
    BRKA_returns = (np.log(BRKA/BRKA.shift()) * 100)
    BRKA_returns.rename(columns={'Adj Close': 'BRKA_returns'}, inplace=True)



    investor = sys.argv[1].lower()
    if "carl" in investor or "icahn" in investor:
        carl_icahn(sys.argv[2], sys.argv[3])
        sys.stdout.flush()

    if "warren" in investor or "buffet" in investor:
        warren_buffett(sys.argv[2], sys.argv[3])
        sys.stdout.flush()

    if "ray" in investor or "dalio" in investor:
        print("Need data on Ray's 'Pure Alpha' and 'All Weather' portfolios")
        sys.stdout.flush()


