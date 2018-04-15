import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pymc3.distributions.timeseries import GaussianRandomWalk
from pylab import rcParams
import pymc3 as pm
import sys
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm

def tinvlogit(x):
    import theano.tensor as t
    return t.exp(x) / (1 + t.exp(x))

def calc_posterior_analytical(data, x, mu_0, sigma_0):
    sigma = 1.
    n = len(data)
    mu_post = (mu_0 / sigma_0**2 + data.sum() / sigma**2) / (1. / sigma_0**2 + n / sigma**2)
    sigma_post = (1. / sigma_0**2 + n / sigma**2)**-1
    return norm(mu_post, np.sqrt(sigma_post)).pdf(x)


def main(sector):
    data = pd.read_csv('sectorAnalysis/DifferencedAvg.csv')
    numStocks = len(data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledData = scaler.fit_transform(data.values).transpose()

    with pm.Model() as mod:
        sigma = pm.Uniform('sigma', float(sigmaMin), float(sigmaMax))
        sigmab = pm.Uniform('sigmab', float(sigmabMin), float(sigmabMax))
        betaPop0 = pm.Normal('betaPop0', mu=0, sd=100)
        beta_0 = pm.Normal('beta_0', mu=betaPop0, sd=sigmab, shape=numStocks)

        x = GaussianRandomWalk('x', sd=sigma, init=pm.Normal.dist(mu=0.0, sd=0.01), shape=numStocks)
        pm.Deterministic('p', tinvlogit(x + betaPop0))
        #
        # for stock in range(numStocks):
        #     stp = 'p{0}'.format(stock)
        #     stn = 'n{0}'.format(stock)
        #     pn = pm.Deterministic(stp, tinvlogit(x + beta_0[stock]))
        #     # pm.Binomial(stn, p=pn, n=np.asarray(data_denom[rat:(rat+1)]), observed=np.asarray(data_numAll[rat:(rat+1)]))
        #     pm.Normal(0,1.0)

    with mod:
        step1 = pm.NUTS(vars=[x, sigmab, beta_0], gamma=.25)
        start2 = pm.sample(2000, step1)[-1]

        # Start next run at the last sampled position.
        step2 = pm.NUTS(vars=[x, sigmab, beta_0], scaling=start2, gamma=.55)
        trace1 = pm.sample(5000, step2, start=start2, progressbar=True)

    summary_dataset = np.percentile(trace1['p'], [5, 50, 95], axis=0)
    with open('sectorAnalysis/{0}DATASET.txt'.format(sector), 'w') as data:
        data.write(str(np.asarray(summary_dataset)))




if __name__ == "__main__":


    stocks = pd.read_csv('stocks.csv')
    sectors = stocks['Sector'].unique()

    sigmaMin = 0.01
    sigmaMax = 10.0
    sigmabMin = 0.01
    sigmabMax =10.0

    for sec in sectors:
        main(sec)