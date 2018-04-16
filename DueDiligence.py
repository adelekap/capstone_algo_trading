from datetime import date
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, TimeDistributed
import numpy as np
from LSTM import reshape, pad
from keras.utils import to_categorical


quarters = {1: "Q4", 2: "Q4", 3: "Q4",
            4: "Q1", 5: "Q1", 6: "Q1",
            7: "Q2", 8: "Q2", 9: "Q2",
            10: "Q3", 11: "Q3", 12: "Q3"}

def to_cat(y_int):
    y_binary = to_categorical(y_int)
    return pd.DataFrame(y_binary)

class sectorSuggestor():
    def __split_data(self,start):
        X = pd.read_csv('sectorAnalysis/SectorData/DifferencedAvg.csv').iloc[:,1:]
        y = to_cat(pd.read_csv('sectorAnalysis/SectorData/ys.csv').iloc[:,2])
        # reshape input to be 3D [samples, timesteps, features]
        Xtrain = reshape(X.iloc[:start,:])
        Xtest = reshape(X.iloc[start:-10,:])
        ytrain = reshape(y[:start])
        ytest = reshape(y[start:])
        return Xtrain, ytrain, Xtest, ytest

    def __init__(self,startdayIndex:int):
        self.startDay = startdayIndex
        self.Xtrain,self.ytrain,self.Xtest,self.ytest = self.__split_data(startdayIndex)
        self.model = Sequential()

    def __get_quarter(self,date: date) -> str:
        month = date.month
        return quarters[month] + " " + str(date.year if quarters[month] != "Q4" else date.year - 1)

    def dev_test(self):
        self.build_sector_NN()
        # paddedSequence = pad(self.Xtest, self.Xtrain.shape[1])
        raw_predictions = self.model.predict(self.Xtest)
        # predictions = undifference(self.rawData.iloc[self.split, 4], unscaled_predictions)


    def build_sector_NN(self,epochs=50):
        self.model.add(LSTM(30,input_shape=(self.Xtrain.shape[1], self.Xtrain.shape[2]),return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(TimeDistributed(Dense(11,activation='softmax')))
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.Xtrain, self.ytrain, epochs=epochs, batch_size=10,verbose=False)
        print(self.model.predict(pad(self.Xtest,self.Xtrain.shape[1])))

    def predict_sector(self,avg):
        pass


if __name__ == '__main__':
    sMod = sectorSuggestor(1000)
    sMod.dev_test()
