
from flask import Flask, request

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import datetime


def carl_icahn(start='2012-01-01', end='2017-01-01'):

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



PERIOD = 'm'
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
# join dataset with Fama-French Factors
IEP_data1 = IEP_returns.join(SP500_excess_returns, how='inner')
IEP_data3 = IEP_returns.join(ffdat, how='inner')[['IEP_returns', 'MKT_PREM', 'SMB', 'HML', 'RF']]
# subtract risk-free from portfolio returns
IEP_data1['IEP_excess_returns'] = IEP_data1['IEP_returns'] - IEP_data3['RF']
IEP_data3['IEP_excess_returns'] = IEP_data3['IEP_returns'] - IEP_data3['RF']


##### Berkshire Hathaway
BRKA.set_index("Date", inplace=1)
BRKA_returns = (np.log(BRKA/BRKA.shift()) * 100)
BRKA_returns.rename(columns={'Adj Close': 'BRKA_returns'}, inplace=True)
# join dataset with Fama-French Factors
BRKA_data1 = BRKA_returns.join(SP500_excess_returns, how='inner')
BRKA_data3 = BRKA_returns.join(ffdat, how='inner')[['BRKA_returns', 'MKT_PREM', 'SMB', 'HML', 'RF']]
# subtract risk-free from portfolio returns
BRKA_data1['BRKA_excess_returns'] = BRKA_data1['BRKA_returns'] - BRKA_data3['RF']
BRKA_data3['BRKA_excess_returns'] = BRKA_data3['BRKA_returns'] - BRKA_data3['RF']





application = Flask(__name__)
application.debug = True
# application.run(host='0.0.0.0')

@application.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        investor = request.form['investor']
        try:
            start = request.form['start']
            end = request.form['end']
        except:
            start = '2000-01-01'
            end = '2016-12-30'

        if "carl" in investor or "icahn" in investor:
            return carl_icahn(start, end)

        if "warren" in investor or "buffet" in investor:
            return warren_buffett(start, end)

        if "ray" in investor or "dalio" in investor:
            return "Need data on Ray's 'Pure Alpha' and 'All Weather' portfolios"

        return "investor name must be either 'carl icahn', 'warren buffett', or 'ray dalio'"

    else:
        return "YOUR REQUEST: GET"



if __name__=="__main__":
    application.run()


