from sklearn.svm import SVR
from DataHandler.putAndGetData import create_timeseries, get_multifeature_data
from sklearn.metrics import mean_squared_error
from utils import diff_multifeature
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from Models.LSTM import undifference


class SVM(object):
    def __split(self):
        """
        Splits data into training and test data for the SVM
        :return: Xtrain, ytrain, Xtest, ytest
        """
        Xtrain = self.scaledData[:self.startDay]
        ytrain = pd.DataFrame(self.scaledData[1:self.startDay + 1]).iloc[:, 4]
        Xtest = self.scaledData[self.startDay:-1]
        ytest = pd.DataFrame(self.scaledData[self.startDay + 1:]).iloc[:, 4]
        return Xtrain, ytrain.values, Xtest, ytest.values

    def __init__(self, C, epsilon, ticker, manager, startDay, kernel='rbf'):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = SVR(C=C, epsilon=epsilon, kernel=kernel)
        self.ticker = ticker
        self.manager = manager
        self.startDay = startDay
        self.data = get_multifeature_data(manager, ticker)
        self.stationaryData = diff_multifeature(self.data)
        self.scaledData = self.scale(self.stationaryData)
        self.Xtrain, self.ytrain, self.Xtest, self.ytest = self.__split()
        self.raw_prices = list(self.data['vwap'])

    def scale(self, df):
        """
        Normalizes the data between 0 and 1
        :param df: dataframe
        :return: scaled dataframe
        """
        values = df.values
        scaled = self.scaler.fit_transform(values)
        return scaled

    def unscale(self, series):
        """
        Unnormalizes the data from the output
        :param series: series of scaled points
        :return: unscaled series
        """
        padded = pd.DataFrame()
        reshaped = series.reshape(1, len(series))[0]
        for i in range(4):
            padded[i] = [0 for j in range(len(series))]
        padded['unscaled'] = reshaped
        padded[5] = [0 for j in range(len(series))]
        unscaled = pd.DataFrame(self.scaler.inverse_transform(padded.values))
        unscaled = unscaled.iloc[:, 4]
        return list(unscaled)

    def test_and_error(self):
        """
        Used in development for deciding the architecture
        :return: None
        """
        self.fit()
        raw_predictions = self.model.predict(self.Xtest)
        unscaled_predictions = self.unscale(raw_predictions)
        predictions = undifference(self.data.iloc[self.startDay, 4], unscaled_predictions)
        print(mean_squared_error(self.ytest, raw_predictions))

        days = create_timeseries(self.manager, self.ticker)[1]
        days = [days[x] for x in range(0, len(days), 2)]
        actual = list(self.data['vwap'])

        plt.plot(days, actual, color='black', label='Actual')
        plt.plot(days[self.startDay + 3:], predictions[1:], color='red', label='LSTM predictions')
        plt.xlabel('day')
        plt.title(self.ticker)
        plt.ylabel('price')
        plt.legend(loc=2)
        plt.savefig('plots/SVM/SVM_{0}_predictions.pdf'.format(self.ticker))
        plt.show()

    def fit(self):
        """
        Trains the model
        :return: None
        """
        self.model.fit(self.Xtrain, self.ytrain)

    def predict(self, D):
        """
        Predicts the next price
        :param D: day index
        :return: prediction
        """
        d = D - len(self.Xtrain) - 1

        if d == -1:
            x = self.Xtrain[len(self.Xtrain) - 1].reshape(1, 6)
        else:
            x = self.Xtest[d].reshape(1, 6)
        previousPrice = self.raw_prices[D - 1]
        diff_pred = self.unscale(self.model.predict(x))
        prediction = previousPrice + diff_pred[0]
        return prediction
