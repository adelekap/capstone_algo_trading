from matplotlib import pyplot as plt
import pandas as pd
import pandas.tools.plotting as pdplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_pacf


class ArimaModel(object):
    def __init__(self,p,d,q,ticker):
        self.p = p
        self.d = d
        self.q = q
        self.ticker = ticker

    def plot_timeseries(self,series):
        plt.plot(range(1, len(series) + 1), series)
        plt.title(self.ticker)
        plt.show()

    def plot_autocorr(self,series):
        pdplot.autocorrelation_plot(series)
        plt.title(self.ticker + ' Autocorrelation Plot')
        plt.savefig('plots/ARIMA/{0}Autocor.pdf'.format(self.ticker))
        plt.close()

    def plot_partial_autocorr(self,series):
        plot_pacf(series, lags=50)
        plt.title(self.ticker + ' Partial Autocorrelation')
        plt.savefig('plots/ARIMA/{0}ParAutocor.pdf'.format(self.ticker))
        plt.close()

    def plot_residuals(self,series):
        model = ARIMA(series, order=(self.p, self.d, self.q))
        model_fit = model.fit(disp=0)
        print(model_fit.summary())

        residuals = pd.DataFrame(model_fit.resid)
        residuals.plot()
        plt.title(self.ticker + ' ARIMA residuals')
        plt.savefig('plots/ARIMA/{0}Resid_{1}{2}{3}.pdf'.format(self.ticker,self.p,self.d,self.q))
        plt.close()
        print(residuals.describe())

    def fit(self,train):
        model = ARIMA(train, order=(self.p, self.d, self.q))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        nextDaysPred = output[0][0]
        return nextDaysPred

    def fit_and_plot(self, series, dates, window=0):
        size = int(len(series) * 0.66)
        train, test = series[0:size], series[size:len(series)]
        predictions = list()
        w = 0
        for t in range(len(test) + 1):
            model = ARIMA(train, order=(self.p, self.d, self.q))
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
        testDates = dates[size:]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1
        ax3 = ax1
        ax1.plot(dates[:size + 1], train[:size + 1], label='Training data', color='black')
        ax2.plot(testDates, test, label='Test data', color='blue')
        ax3.plot(testDates, predictions[:len(testDates)], color='red', label='ARIMA Predictions')
        plt.legend()
        plt.text(min(dates), max(train), 'Test MSE = ' + str(error))
        plt.title('{0} ARIMA({1},{2},{3})'.format(self.ticker, self.p, self.d, self.q))
        plt.savefig('plots/ARIMA/{0}Arima.pdf'.format(self.ticker))
        plt.close()
