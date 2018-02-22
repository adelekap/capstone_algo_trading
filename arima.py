from matplotlib import pyplot as plt
import pandas as pd
import pandas.tools.plotting as pdplot
from mongoObjects import CollectionManager
from pymongo import MongoClient
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_pacf
from putAndGetData import create_timeseries


def parameterize_arima(series, ticker,p,d,q):
    """
    Used to determine the order of the ARIMA.
    Plots the data, autocorrelation plot,
    partial autocorrelation plot, and residuals.
    :param series: timeseries
    :param ticker: string of ticker
    :return: None
    """
    # Plot time series
    plt.plot(range(1, len(series) + 1), series)
    plt.title(ticker)
    plt.close()

    # Plot autocorrelation plot
    pdplot.autocorrelation_plot(series)
    plt.title(ticker + ' Autocorrelation Plot')
    plt.savefig('plots/ARIMA/{0}Autocor.pdf'.format(ticker))
    plt.close()

    # Plot Partial Autocorrelation
    plot_pacf(series, lags=50)
    plt.title(ticker+' Partial Autocorrelation')
    plt.savefig('plots/ARIMA/{0}ParAutocor.pdf'.format(ticker))
    plt.close()

    # Fit Model
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    # Plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.title(ticker + ' ARIMA residuals')
    plt.savefig('plots/ARIMA/{0}Resid_{1}{2}{3}.pdf'.format(ticker,p,d,q))
    plt.close()
    print(residuals.describe())


def fit_arima(series, p, d, q,dates,window=0,plot=True):
    """
    Fits the timeseries data to an ARIMA model.
    :param series: timeseries data
    :param p: lag order
    :param d: differencing order
    :param q: size of moving average window
    :return: MSE
    """
    size = int(len(series) * 0.66)
    train, test = series[0:size], series[size:len(series)]
    predictions = list()
    w = 0
    for t in range(len(test)+1):
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0][0]
        predictions.append(yhat)
        obs = yhat
        if w == window and t != len(test):
            train.append(test[t])
            w = 0
        else:
            train.append(obs)
            w += 1
    error = mean_squared_error(test, predictions[:len(test)])

    # plot
    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1
        ax3 = ax1
        testDates = dates[size:]
        ax1.plot(dates[:size+1],train[:size+1],label='Training data',color='black')
        ax2.plot(testDates,test, label='Test data',color = 'blue')
        ax3.plot(testDates,predictions[:len(testDates)], color='red', label='ARIMA Predictions')
        plt.legend()
        plt.text(min(dates),max(train), 'Test MSE = ' + str(error))
        plt.title('{0} ARIMA({1},{2},{3})'.format(ticker,p,d,q))
        plt.savefig('plots/ARIMA/{0}Arima.pdf'.format(ticker))
        plt.close()
        print(predictions[len(testDates)])  # Next day's prediction
    return error


if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
    ticker = 'googl'

    series,dates = create_timeseries(manager,ticker)

    ar = 1  # base predictions on last day
    i = 1  # single differencing
    ma = 0  # only autoregressive components
    parameterize_arima(series, ticker,ar,i,ma)
    fit_arima(series,ar,i,ma,dates)
