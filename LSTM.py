from putAndGetData import get_multifeature_data, create_timeseries
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import LSTM, Dense, Flatten, Dropout, Input, TimeDistributed, Reshape
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend
from utils import diff_multifeature
import numpy as np
from sklearn.metrics import mean_squared_error


def reshape(df, y=False):
    """
    Reshapes the input to the LSTM
    :param df: dataframe
    :param y: if this is the response variable
    :return: reshaped values
    """
    values = df.values
    # reshape input to be 3D [samples, timesteps, features]
    if y:
        return values.reshape((1, values.shape[0], 1))
    # return values.reshape((1,values.shape[0], values.shape[1]))
    return values.reshape((1, values.shape[0], values.shape[1]))


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def undifference(first, series):
    """
    Undifferences a series
    :param first: the first actual value in the series
    :param series: the series of differences
    :return: undifferenced series
    """
    undifferenced = [first]
    for num in series:
        lastItem = undifferenced[-1]
        undifferenced.append(lastItem + num)
    return undifferenced[1:]


def pad(sequence, maxlen, featN=6):
    """
    Pads a sequence to some length for input into the LSTM
    :param sequence: sequence of values
    :param maxlen: maximum length the sequence needs to be
    :param featN: feature number
    :return: padded sequence
    """
    sequence = list(sequence)
    length = maxlen - len(sequence)
    padding = [[0 for i in range(featN)] for i in range(length)]
    padded_array = np.array([sequence + padding])
    return padded_array


class NeuralNet(object):
    def scale(self, df):
        """
        Normalizes the values between 0 and 1
        :param df: dataframe of data
        :return: scaled numpy array
        """
        values = df.values
        scaled = self.scaler.fit_transform(values)
        return scaled

    def unscale(self, series):
        """
        Unscales the output of the LSTM
        :param series: series of data
        :return: unormalized data
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

    def __get_and_split_data(self, data, split):
        """
        Splits the data into training, validation, and test
        data for the LSTM
        :param data: all data
        :param split: split into training, validation, and test data
        :return: Xtrain, ytrain, Xval, yval, Xtest, ytest
        """
        valPoint = int(split * .85)
        Xtrain = reshape(data[:valPoint])
        ytrain = reshape(data[1:valPoint + 1].iloc[:, 4], True)

        Xval = reshape(data[valPoint:split])
        yval = reshape(data[valPoint + 1:split + 1].iloc[:, 4], True)

        Xtest = reshape(data[split:-1])
        ytest = reshape(data[split + 1:].iloc[:, 4], True)
        return Xtrain, ytrain, Xval, yval, Xtest, ytest

    def __init__(self, ticker, manager, split_point):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.manager = manager
        self.ticker = ticker
        self.rawData = get_multifeature_data(manager, ticker)
        self.differencedData = diff_multifeature(self.rawData)
        self.data = pd.DataFrame(self.scale(self.differencedData))
        self.model = Sequential()
        self.history = None
        self.split = split_point
        self.Xtrain, self.ytrain, self.Xval, self.yval, self.Xtest, self.ytest = self.__get_and_split_data(self.data,
                                                                                                           self.split)

    def log_learning(self, hist=True):
        """For visualizing learning during development"""
        # ./tensorboard --logdir='/Users/adelekap/Documents/capstone_algo_trading/logs' --host localhost
        return TensorBoard(log_dir='./logs', histogram_freq=10, write_grads=hist,
                           write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                           embeddings_metadata=None)

    def test_and_error(self, epochs=500):
        """
        Used in development to determine architecture of LSTM
        :param epochs: number of epochs to train
        :return: None
        """
        self.create_network()
        self.train_network(epochs=epochs, dev=False)
        paddedSequence = pad(self.Xtest[0], self.Xtrain.shape[1])
        raw_predictions = self.model.predict(paddedSequence)[0][:(1257 - self.split)]
        unscaled_predictions = self.unscale(raw_predictions)
        predictions = undifference(self.rawData.iloc[self.split, 4], unscaled_predictions)
        valLossfix = [(self.history.history['val_rmse'][v] + (1 / epochs) * (epochs - v)) for v in range(15)]
        valLoss = valLossfix + self.history.history['val_rmse'][15:]

        print(mean_squared_error(list(self.rawData['vwap'])[self.split + 3:], predictions[1:]))

        plt.plot(self.history.history['rmse'])
        plt.plot(valLoss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.xlim(1, epochs)
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('plots/LSTM/trainTestLoss.pdf')
        plt.show()

        days = create_timeseries(self.manager, self.ticker)[1]
        days = [days[x] for x in range(0, len(days), 2)]
        actual = list(self.rawData['vwap'])

        actualResults = pd.DataFrame()
        predictionResults = pd.DataFrame()

        actualResults['Dates'] = days
        actualResults['Actual'] = actual
        predictionResults['Dates'] = days[self.split + 3:]
        predictionResults['Predictions'] = predictions[1:]
        modelResults = pd.merge(actualResults, predictionResults, on='Dates')
        modelResults.to_csv(
            '/Users/adelekap/Documents/capstone_algo_trading/comparison/LSTM/{0}Results.csv'.format(self.ticker))

        plt.plot(days, actual, color='black', label='Actual')
        plt.plot(days[self.split + 3:], predictions[1:], color='red', label='LSTM predictions')
        plt.xlabel('day')
        plt.title(self.ticker)
        plt.ylabel('price')
        plt.legend(loc=2)
        plt.savefig('plots/LSTM/LSTM_{0}_predictions.pdf'.format(self.ticker))
        plt.show()

    def create_network(self):
        """
        Builds the LSTM network
        :return: None
        """
        self.model.add(LSTM(24, return_sequences=True, input_shape=(self.Xtrain.shape[1], self.Xtrain.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(TimeDistributed(Dense(1, activation='tanh')))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])

    def train_network(self, epochs=500, dev=False):
        """
        Trains the network
        :param epochs: Number of epochs
        :param dev: If it is in development (for logging learning)
        :return: None
        """
        validationX = pad_sequences(self.Xval, maxlen=self.Xtrain.shape[1])
        validationy = pad_sequences(self.yval, maxlen=self.ytrain.shape[1])
        self.history = self.model.fit(self.Xtrain, self.ytrain, epochs=epochs, batch_size=1, verbose=dev,
                                      validation_data=(validationX, validationy))

    def predict(self, D):
        """
        Predicts the next value in the series
        :param D: day index
        :return: prediction
        """
        d = D - (self.Xtrain.shape[1] + self.Xval.shape[1])
        if d == 0:
            x = [self.Xtrain[0][-1]]
        else:
            x = [self.Xtest[0][d - 1]]
        x = pad(x, self.Xtrain.shape[1])
        raw_prediction = self.model.predict(x)[0][0]
        unscaled_prediction = self.unscale(raw_prediction)
        prediction = undifference(self.rawData.iloc[self.split, 4], unscaled_prediction)[0]
        return prediction
