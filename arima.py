from matplotlib import pyplot as plt
import pandas as pd
import pandas.tools.plotting as pdplot
from mongoObjects import CollectionManager
from pymongo import MongoClient
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_pacf
import datetime as dt


def create_timeseries(manager, ticker):
    """
    Creates a timeseries of a chain of opening
    and closing prices for specified stock.
    :param manager: collection manager
    :param ticker: string of ticker
    :return: timeseries
    """
    mmData = manager.find({'ticker': ticker})
    series = []
    dates = []
    for index, row in mmData.iterrows():
        series.append(row['open'])
        series.append(row['close'])
        dates.append(dt.datetime.strptime(row['date']+'-09',"%Y-%m-%d-%H"))
        dates.append(dt.datetime.strptime(row['date']+'-04',"%Y-%m-%d-%H"))
    return series,dates


def parameterize_arima(series, ticker):
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
    plt.show()

    # Plot autocorrelation plot
    pdplot.autocorrelation_plot(series)
    plt.title(ticker + ' Autocorrelation Plot')
    plt.savefig(ticker+'Autocor.png')
    plt.show()

    # Plot Partial Autocorrelation
    plot_pacf(series, lags=50)
    plt.title(ticker+' Partial Autocorrelation')
    plt.savefig(ticker+'ParAutocor.png')
    plt.show()

    # Fit Model
    model = ARIMA(series, order=(2, 1, 0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    # Plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.title(ticker + ' ARIMA residuals')
    plt.show()
    residuals.plot(kind='kde')
    plt.title(ticker + ' ARIMA residual density plot')
    plt.savefig(ticker+'Resid.png')
    plt.show()
    print(residuals.describe())


def fit_arima(series, p, d, q,dates):
    """
    Fits the timeseries data to an ARIMA model.
    :param series: timeseries data
    :param p: lag order
    :param d: differencing order
    :param q: size of moving average window
    :return:
    """
    size = int(len(series) * 0.66)
    train, test = series[0:size], series[size:len(series)]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        # obs = test[t]
        obs = yhat
        train.append(obs)
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot
    testDates = dates[size:]
    plt.plot(dates,train,label='Training data',color='black')
    plt.plot(testDates,test, label='Test data',color = 'blue')
    plt.plot(testDates,predictions, color='red', label='ARIMA Predictions')
    plt.legend(loc=4)
    plt.text(min(dates),max(train), 'Test MSE = ' + str(error))
    plt.title('{0} ARIMA({1},{2},{3})'.format(ticker,p,d,q))
    plt.xlim(testDates[0],testDates[len(testDates)])
    plt.savefig(ticker+'Arima.pdf')
    plt.show()


if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
    ticker = 'gd'
    test = create_timeseries(manager,ticker)

    series,dates = create_timeseries(manager,ticker)
    parameterize_arima(series,ticker)
    fit_arima(series,2,1,0,dates)