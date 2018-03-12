from putAndGetData import create_timeseries
from mongoObjects import CollectionManager,MongoClient
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class NN(object):
    def __transform_data(self,data:list):
        trainIndex = int(0.75 * len(data))
        diff_values = self.difference(data)  # Make timeseries stationary
        train = self.timeseries_to_supervised(diff_values[:trainIndex+1])
        test = self.timeseries_to_supervised(diff_values[trainIndex:])
        train_scaled, test_scaled = self.scale(train.values,test.values)
        return train_scaled, test_scaled

    def __init__(self,data:list):
        self.raw_data = data
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.train, self.test = self.__transform_data(data)
        self.model = None


    def timeseries_to_supervised(self,data:list,lag=1):
        """
        Makes the list of prices into a supervised (labeled) dataset.
        X = price for the day
        Y = what the prediction should be for the next day
        :param data: list of prices for the ticker
        :return: supervised dataset (as a dataframe)
        """
        df = pd.DataFrame()
        df['X'] = data
        df['Y'] = data[1:]+[None]
        df = df[:-1]
        return df

    def difference(self,data:list):
        diff = list()
        for i in range(1, len(data)):
            value = data[i] - data[i - 1]
            diff.append(value)
        return diff

    def invert(self,history:list,yhat:float,interval=1):
        return yhat + history[-interval]

    def scale(self,train, test):
        # fit scaler
        self.scaler = self.scaler.fit(train)
        # transform train
        train = train.reshape(train.shape[0], train.shape[1])
        train_scaled = self.scaler.transform(train)
        # transform test
        test = test.reshape(test.shape[0], test.shape[1])
        test_scaled = self.scaler.transform(test)
        return train_scaled, test_scaled

    def invert_scale(self, X, value):
        new_row = [x for x in X] + [value]
        array = np.array(new_row)
        array = array.reshape(1, len(array))
        inverted = self.scaler.inverse_transform(array)
        return inverted[0, -1]

    def fit(self,batch_size, nb_epoch, neurons):
        X, y = self.train[:, 0:-1], self.train[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        for i in range(nb_epoch):
            model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
            model.reset_states()
        self.model = model

    def forecast(self, batch_size, X):
        # A one step forecast
        X = X.reshape(1, 1, len(X))
        yhat = self.model.predict(X, batch_size=batch_size)
        return yhat[0, 0]

    def predict(self):
        train_reshaped = self.train[:, 0].reshape(len(self.train), 1, 1)
        self.model.predict(train_reshaped, batch_size=1)

        # walk-forward validation on the test data
        predictions = list()
        exp = list()
        for i in range(len(self.test)):
            # make one-step forecast
            X, y = self.test[i, 0:-1], self.test[i, -1]
            yhat = self.forecast(1, X)
            # invert scaling
            yhat = self.invert_scale(X, yhat)
            # invert differencing
            yhat = self.invert(self.raw_data, yhat, len(self.test) + 1 - i)
            # store forecast
            predictions.append(yhat)
            expected = self.raw_data[len(self.train) + i + 1]
            exp.append(expected)
            # print('Predicted={0}, Expected={1}'.format(yhat, expected))
        return predictions,exp





if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
    data = create_timeseries(manager,'aapl')[0]

    network = NN(data)
    network.fit(1, 10, 4)
    predictions,expected = network.predict()

    rmse = sqrt(mean_squared_error(expected, predictions))
    plt.plot(predictions)
    plt.plot(expected)
    plt.show()

    print('Test RMSE: %.3f' % rmse)
