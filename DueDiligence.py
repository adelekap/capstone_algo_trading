from datetime import date, datetime
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, TimeDistributed, Flatten
import numpy as np
from LSTM import reshape, pad, rmse
from keras.utils import to_categorical
import operator
from sklearn.metrics import accuracy_score
from addFundamentals import get_all_fundamentals
import random
from sklearn.preprocessing import LabelEncoder

sector_to_id = {'Industrials': 0, 'Health Care': 1, 'Information Technology': 2, 'Consumer Discretionary': 3,
                'Utilities': 4, 'Financials': 5, 'Materials': 6, 'Consumer Staples': 7, 'Real Estate': 8,
                'Energy': 9, 'Telecommunications Services': 10}

stocks = pd.read_csv('stocks.csv')
sectors = stocks['Sector'].unique()


def stocks_in_sector(sector):
    stocksInSector = stocks[stocks['Sector'] == sector]
    return list(stocksInSector['Symbol'])


def to_cat(y_int):
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
        self.build_sector_NN(10)
        paddedtestData = pad(self.Xtest[0], self.Xtrain.shape[1], 11)
        predictions = self.__to_sector(self.model.predict(paddedtestData)[0][:self.ytest.shape[1]])
        # print(accuracy_score(self.targets,predictions))

    def build_sector_NN(self, epochs=20):
        self.model.add(LSTM(33, input_shape=(self.Xtrain.shape[1], self.Xtrain.shape[2]), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(TimeDistributed(Dense(11, activation='softmax')))
        self.model.compile(optimizer="nadam", loss='categorical_crossentropy', metrics=['accuracy'])
        print('TRAINING SECTOR SUGGESTOR NETWORK')
        self.history = self.model.fit(self.Xtrain, self.ytrain, epochs=epochs, batch_size=10, verbose=False)

    def predict_sector(self, D):
        d = D - (self.Xtrain.shape[1])
        X = [self.Xtest[0][d]]
        paddedtestData = pad(X, self.Xtrain.shape[1], 11)
        prediction = self.__to_sector(self.model.predict(paddedtestData)[0][:self.ytest.shape[1]])[0]
        return prediction


class StockSuggestor():
    def __train_and_test(self):
        Xtrain, ytrain_raw, test,testDates,unionStocks = get_all_fundamentals(self.stocks,self.tradeDay)
        encoder = LabelEncoder()
        encoder.fit(unionStocks)
        encoded_Y = encoder.transform(ytrain_raw)
        ytrain = to_categorical(encoded_Y,len(unionStocks))
        self.stocks = unionStocks
        self.testDates = testDates
        return Xtrain, ytrain, test

    def __init__(self, sector: str, dayIndex, dayString):
        self.testDates = None
        self.sector = sector
        self.startIndex = dayIndex
        self.stocks = stocks_in_sector(sector)
        self.tradeDay = datetime.strptime(dayString, '%Y-%m-%d').date()
        self.Xtrain, self.ytrain, self.test = self.__train_and_test()
        self.model = Sequential()

    def build_network(self, epochs=50):
        self.model.add(Dense(24, activation='relu', input_shape=(self.Xtrain.shape[1],self.Xtrain.shape[2])))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(len(self.stocks), activation='softmax'))
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())
        print('TRAINING STOCK SUGGESTOR NETWORK')
        self.history = self.model.fit(self.Xtrain, self.ytrain, epochs=epochs, batch_size=10, verbose=False)

    def predict_stock(self, dayString):
        todaysDate = datetime.strptime(dayString,'%Y-%m-%d')
        testD = 0
        for d in self.testDates:
            if d < todaysDate:
                testD += 1
        lastQuartersFundamentals = self.test[testD]
        prediction = self.model.predict(lastQuartersFundamentals)
        print(prediction)
        return prediction


if __name__ == '__main__':
    sMod = SectorSuggestor(1000)
    sMod.dev_test()
