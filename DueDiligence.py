from datetime import date
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, TimeDistributed
import numpy as np
from LSTM import reshape, pad, rmse
from keras.utils import to_categorical
import operator
from sklearn.metrics import accuracy_score


def to_cat(y_int):
    y_binary = to_categorical(y_int)
    return pd.DataFrame(y_binary)


class sectorSuggestor():
    def __to_sector(self,predictions):
        sectors = []
        for pred in predictions:
            max_index, max_value = max(enumerate(pred), key=operator.itemgetter(1))
            sectors.append(max_index)
        return sectors

    def __split_data(self,start):
        X = pd.read_csv('sectorAnalysis/SectorData/DifferencedAvg.csv').iloc[:,1:]
        y = pd.read_csv('sectorAnalysis/SectorData/ys.csv').iloc[:,2]
        ytarget = list(y[start:])
        y = to_cat(y)
        Xtrain = reshape(X.iloc[:start,:])
        Xtest = reshape(X.iloc[start:-10,:])
        ytrain = reshape(y[:start])
        ytest = reshape(y[start:])
        return Xtrain, ytrain, Xtest, ytest,ytarget

    def __init__(self,startdayIndex:int):
        self.startDay = startdayIndex
        self.Xtrain,self.ytrain,self.Xtest,self.ytest,self.targets = self.__split_data(startdayIndex)
        self.model = Sequential()
        self.history = None

    def dev_test(self):
        self.build_sector_NN(10)
        paddedtestData = pad(self.Xtest[0], self.Xtrain.shape[1],11)
        predictions = self.__to_sector(self.model.predict(paddedtestData)[0][:self.ytest.shape[1]])
        # print(accuracy_score(self.targets,predictions))

    def build_sector_NN(self,epochs=50):
        self.model.add(LSTM(40,input_shape=(self.Xtrain.shape[1], self.Xtrain.shape[2]),return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(TimeDistributed(Dense(11,activation='softmax')))
        self.model.compile(optimizer="nadam", loss='categorical_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit(self.Xtrain, self.ytrain, epochs=epochs, batch_size=10,verbose=False)

    def predict_sector(self,D):
        d = D - (self.Xtrain.shape[1] + self.Xval.shape[1])
        X = self.Xtest[d]
        paddedtestData = pad(X, self.Xtrain.shape[1], 11)
        prediction = self.__to_sector(self.model.predict(paddedtestData)[0][:self.ytest.shape[1]])
        return prediction


if __name__ == '__main__':
    sMod = sectorSuggestor(1000)
    sMod.dev_test()