from putAndGetData import create_timeseries
from mongoObjects import CollectionManager,MongoClient
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np


class NN(object):
    #Todo: Add set for RSME before predictions
    def __transform_data(self,data:list,trainPercent):
        trainIndex = int(trainPercent * len(data))
        self.rawTrain = data[:trainIndex]
        self.rawTest = data[trainIndex:]
        diff_values = self.difference(data)  # Make timeseries stationary
        train = self.timeseries_to_supervised(diff_values[:trainIndex+1])
        test = self.timeseries_to_supervised(diff_values[trainIndex:])
        train_scaled, test_scaled = self.scale(train.values,test.values)
        return train_scaled, test_scaled

    def __init__(self,data:list,trainPercent=0.75):
        self.raw_data = data
        self.rawTrain = None
        self.rawTest = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.train, self.test = self.__transform_data(data,trainPercent)
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
        """
        Differences the timeseries so it is stationary
        :param data: list of timeseries points
        :return: differenced timeseries
        """
        diff = list()
        for i in range(1, len(data)):
            value = data[i] - data[i - 1]
            diff.append(value)
        return diff

    def invert(self,history:list,yhat:float,interval=1):
        return yhat + history[-interval]

    def scale(self,train, test):
        """
        Scale the data for the NN
        :param train:
        :param test:
        :return:
        """
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
        """
        Revert back to the original scale.
        :param X: Scaled prices
        :param value:
        :return: unscaled prices
        """
        new_row = [x for x in X] + [value]
        array = np.array(new_row)
        array = array.reshape(1, len(array))
        inverted = self.scaler.inverse_transform(array)
        return inverted[0, -1]

    def fit_lstm(self,batch_size, nb_epoch, neurons):
        """
        Trains the LSTM
        :param batch_size: size of minibatch
        :param nb_epoch: number of epochs
        :param neurons: number of neurons
        :return: None
        """
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
        train_reshaped = self.train[:, 0].reshape(len(self.train), 1, 1)
        self.model.predict(train_reshaped, batch_size=1)

    def fit(self, X:list):
        """
        A one step forecast into the future.
        """
        X = np.array([X])
        X = X.reshape(1, 1, len(X))
        yhat = self.model.predict(X, batch_size=1)[0,0]
        yhat = self.invert_scale(X, yhat)
        yhat = self.invert(self.raw_data, yhat, len(self.test) + 1 - 0)
        return yhat



if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', 'AlgoTradingDB')
    data = create_timeseries(manager,'hal')[0]

    network = NN(data)
    network.fit_lstm(1, 10, 4)
    print(np.mean(data))

    # point = network.test[0, 0:-1]
    test = np.array([50])
    prediction = network.fit(test)

    print(prediction)
