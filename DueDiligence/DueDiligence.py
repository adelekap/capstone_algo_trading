from datetime import date, datetime
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, TimeDistributed, Flatten
import numpy as np
from Models.LSTM import reshape, pad, rmse
from keras.utils import to_categorical
import operator
from DueDiligence.addFundamentals import get_all_fundamentals
import random
from sklearn.preprocessing import LabelEncoder

sector_to_id = {'Industrials': 0, 'Health Care': 1, 'Information Technology': 2, 'Consumer Discretionary': 3,
                'Utilities': 4, 'Financials': 5, 'Materials': 6, 'Consumer Staples': 7, 'Real Estate': 8,
                'Energy': 9, 'Telecommunications Services': 10}

stocks = pd.read_csv('stocks.csv')
sectors = stocks['Sector'].unique()


def stocks_in_sector(sector: str):
    """
    Returns a list of stocks in a given sector
    :param sector: sector
    :return: list of stocks
    """
    stocksInSector = stocks[stocks['Sector'] == sector]
    return list(stocksInSector['Symbol'])


def to_cat(y_int):
    """
    Converts a list of categories to 1-hot encoded data
    :param y_int: list of categories
    :return: dataframe of the vectors
    """
    y_binary = to_categorical(y_int)
    return pd.DataFrame(y_binary)


class SectorSuggestor():
    def __to_sector(self, predictions):
        sectors = []
        for pred in predictions:
            max_index, max_value = max(enumerate(pred), key=operator.itemgetter(1))
            sectors.append(max_index)
        return sectors

    def __split_data(self, start):
        """
        Splits data into training and testing for the sector suggestor
        :param start: start day
        :return: Xtrain, ytrain, Xtest, ytest, ytarget values
        """
        X = pd.read_csv('sectorAnalysis/SectorData/DifferencedAvg.csv').iloc[:, 1:]
        y = pd.read_csv('sectorAnalysis/SectorData/ys.csv').iloc[:, 2]
        ytarget = list(y[start:])
        y = to_cat(y)
        Xtrain = reshape(X.iloc[:start, :])
        Xtest = reshape(X.iloc[start:-10, :])
        ytrain = reshape(y[:start])
        ytest = reshape(y[start:])
        return Xtrain, ytrain, Xtest, ytest, ytarget

    def __init__(self, startdayIndex: int):
        random.seed(100)
        self.startDay = startdayIndex
        self.Xtrain, self.ytrain, self.Xtest, self.ytest, self.targets = self.__split_data(startdayIndex)
        self.model = Sequential()
        self.history = None

    def dev_test(self):
        """
        Used in development to decide the architecture of the network
        :return: None
        """
        self.build_sector_NN(10)
        paddedtestData = pad(self.Xtest[0], self.Xtrain.shape[1], 11)
        predictions = self.__to_sector(self.model.predict(paddedtestData)[0][:self.ytest.shape[1]])
        # print(accuracy_score(self.targets,predictions))

    def build_sector_NN(self, epochs=20):
        """
        Builds the model and trains it
        :param epochs: number of epochs to train for
        :return: None
        """
        self.model.add(LSTM(33, input_shape=(self.Xtrain.shape[1], self.Xtrain.shape[2]), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(TimeDistributed(Dense(11, activation='softmax')))
        self.model.compile(optimizer="nadam", loss='categorical_crossentropy', metrics=['accuracy'])
        print('TRAINING SECTOR SUGGESTOR NETWORK')
        self.history = self.model.fit(self.Xtrain, self.ytrain, epochs=epochs, batch_size=10, verbose=False)

    def predict_sector(self, D):
        """
        Predicts the next best sector in the series
        :param D: day index
        :return: prediction
        """
        d = D - (self.Xtrain.shape[1])
        X = [self.Xtest[0][d]]
        paddedtestData = pad(X, self.Xtrain.shape[1], 11)
        prediction = self.__to_sector(self.model.predict(paddedtestData)[0][:self.ytest.shape[1]])[0]
        return prediction


class StockSuggestor():
    def __train_and_test(self):
        """
        Splits the data into training and test data
        :return: Xtrain, ytrain, test data
        """
        Xtrain, ytrain_raw, test, testDates, unionStocks = get_all_fundamentals(self.stocks, self.tradeDay)
        self.encoder.fit(unionStocks)
        encoded_Y = self.encoder.transform(ytrain_raw)
        ytrain = to_categorical(encoded_Y, len(unionStocks))
        self.stocks = unionStocks
        self.testDates = testDates
        return Xtrain, ytrain, test

    def __init__(self, sector: str, dayIndex, dayString):
        self.encoder = LabelEncoder()
        self.testDates = None
        self.sector = sector
        self.startIndex = dayIndex
        self.stocks = stocks_in_sector(sector)
        self.tradeDay = datetime.strptime(dayString, '%Y-%m-%d').date()
        self.Xtrain, self.ytrain, self.test = self.__train_and_test()
        self.model = Sequential()

    def build_network(self, epochs=50):
        """
        Builds the stock suggestor network and trains it
        :param epochs: Number of epochs to train for
        :return: None
        """
        self.model.add(Dense(24, activation='tanh', input_shape=(self.Xtrain.shape[1], self.Xtrain.shape[2])))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='tanh'))
        self.model.add(Dense(len(self.stocks), activation='softmax'))
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        print('TRAINING STOCK SUGGESTOR NETWORK')
        self.history = self.model.fit(self.Xtrain, self.ytrain, epochs=epochs, batch_size=10, verbose=False)

    def predict_stock(self, dayString):
        """
        Predicts the best stock given quarterly reports
        :param dayString: string of the current day
        :return: predicted stock
        """
        todaysDate = datetime.strptime(dayString, '%Y-%m-%d')
        testD = 0
        for d in self.testDates:
            if d < todaysDate:
                testD += 1
        lastQuartersFundamentals = self.test[testD].reshape(1, len(self.stocks), 11)
        predictionProbs = self.model.predict(lastQuartersFundamentals)
        prediction = [np.argmax(predictionProbs)]
        return self.encoder.inverse_transform(prediction)[0], self.testDates[-1]
