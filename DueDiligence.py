from datetime import date
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
import numpy as np

quarters = {1: "Q4", 2: "Q4", 3: "Q4",
            4: "Q1", 5: "Q1", 6: "Q1",
            7: "Q2", 8: "Q2", 9: "Q2",
            10: "Q3", 11: "Q3", 12: "Q3"}

class sectorSuggestor():
    def __split_data(self,start):
        X = pd.read_csv('sectorAnalysis/SectorData/DifferencedAvg.csv')
        y = pd.read_csv('sectorAnalysis/SectorData/ys.csv')
        # reshape input to be 3D [samples, timesteps, features]
        Xtrain = X.iloc[:start,:].values.reshape(11,len(X),1)
        Xtest = X.iloc[start:,:].values.reshape(11,len(X),1)
        ytrain = y[:start].values
        ytest = y[start:].values
        return Xtrain, ytrain, Xtest, ytest

    def __init__(self,startdayIndex:int):
        self.startDay = startdayIndex
        self.Xtrain,self.ytrain,self.Xtest,self.ytest = self.__split_data(startdayIndex)
        self.model = Sequential()

    def __get_quarter(self,date: date) -> str:
        month = date.month
        return quarters[month] + " " + str(date.year if quarters[month] != "Q4" else date.year - 1)

    def build_sector_NN(self,epochs=50):
        self.model.add(LSTM(30,
                       input_shape=(self.Xtrain.shape[1], 1),
                       return_sequences=True,
                       stateful=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(11))
        self.model.add(Activation("softmax"))
        self.model.compile(optimizer="adam", loss=self.loss, metrics=['accuracy'])
        self.model.fit(self.Xtrain, self.ytrain, epochs=epochs, batch_size=10,verbose=False)

    def predict_sector(self,avg):
        pass

