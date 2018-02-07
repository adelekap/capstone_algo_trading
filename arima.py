from matplotlib import pyplot as plt
import pandas as pd
import pandas.tools.plotting as pdplot
from mongoObjects import CollectionManager
from pymongo import MongoClient
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_pacf


def create_timeseries(manager, ticker):
    mmData = manager.find({'ticker': ticker})
    series = []
    for index, row in mmData.iterrows():
        series.append(row['open'])
        series.append(row['close'])
    return series


def parameterize_arima(series, ticker):
    # Plot time series
    plt.plot(range(1, len(series) + 1), series)
    plt.title(ticker)
    plt.show()

    # Plot autocorrelation plot
    pdplot.autocorrelation_plot(series)
    plt.title(ticker + ' Autocorrelation Plot')
    plt.show()

    # Plot Partial Autocorrelation
    plot_pacf(series, lags=50)
    plt.title(ticker+' Partial Autocorrelation')
    plt.show()

    # Fit Model
    model = ARIMA(series, order=(3, 1, 0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    # Plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.title(ticker + ' ARIMA residuals')
    plt.show()
    residuals.plot(kind='kde')
    plt.title(ticker + ' ARIMA residual density plot')
    plt.show()
    print(residuals.describe())


def fit_arima(series, p, d, q):
    size = int(len(series) * 0.66)
    train, test = series[0:size], series[size:len(series)]
    # history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        train.append(obs)
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot
    plt.plot(test, label='Ground Truth')
    plt.plot(predictions, color='red', label='ARIMA Predictions', alpha=0.8)
    plt.legend()
    plt.text(2, 2, 'Test MSE = ' + str(error))
    plt.savefig('arima.pdf')
    plt.show()


if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
    ticker = 'aapl'
    parameterize_arima(create_timeseries(manager,ticker),ticker)

