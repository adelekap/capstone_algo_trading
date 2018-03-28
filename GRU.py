from putAndGetData import get_multifeature_data
from mongoObjects import CollectionManager,MongoClient
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, GRU
from math import sqrt
from sklearn.metrics import mean_squared_error

def reshape(df):
    values = df.values
    return values.reshape((values.shape[0], 1, values.shape[1]))


class NeuralNet(object):
    def __get_and_split_data(self,data):
        splitPoint = int(len(data) * 0.8)
        Xtrain = reshape(data[:splitPoint])
        ytrain = data[1:splitPoint + 1]['vwap']
        Xtest = reshape(data[splitPoint:-1])
        ytest = data[splitPoint + 1:]['vwap']
        print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)
        return Xtrain,ytrain,Xtest,ytest

    def __init__(self,ticker,manager):
        self.manager = manager
        self.data = get_multifeature_data(self.manager, 'aapl')
        self.Xtrain,self.ytrain,self.Xtest,self.ytest = self.__get_and_split_data(self.data)
        self.model = Sequential()
        self.history = None
        # self.scaler = MinMaxScaler(feature_range=(0, 1))
        # self.scaler = self.scaler.fit_transform(self.data.values)

    def create_network(self):
        print(self.Xtrain.shape)
        self.model.add(GRU(6,input_shape=(self.Xtrain.shape[1], self.Xtrain.shape[2]),
                           activation='tanh',recurrent_activation='relu'))
        self.model.add(Dense(1,activation='relu'))
        self.model.compile(loss='mean_squared_error',optimizer='adam')
        print(self.model.summary())

    def train_network(self,plotting=False):
        self.history = self.model.fit(self.Xtrain,self.ytrain,
                                      epochs=80,verbose=False)

    def predict(self):
        yhat = self.model.predict(self.Xtest)
        print(sqrt(mean_squared_error(self.ytest, yhat)))



if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', 'AlgoTradingDB')
    NN = NeuralNet('aapl',manager)
    NN.create_network()
    NN.train_network()
    NN.predict()

