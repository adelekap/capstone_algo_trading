from putAndGetData import get_multifeature_data
from mongoObjects import CollectionManager,MongoClient
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout, Flatten, LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, TensorBoard

def reshape(df):
    values = df.values
    return values.reshape((values.shape[0], 1, values.shape[1]))

def log_learning(hist=True):
    """For visualizing learning during development"""
    # ./tensorboard --logdir='/Users/adelekap/Documents/capstone_algo_trading/logs' --host localhost
    return TensorBoard(log_dir='./logs',histogram_freq=10, write_grads=hist,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None)


class NeuralNet(object):
    def __get_and_split_data(self,data):
        splitPoint = int(len(data) * 0.8)
        Xtrain = reshape(data[:splitPoint])
        ytrain = data[1:splitPoint + 1]['vwap']
        Xtest = reshape(data[splitPoint:-1])
        ytest = data[splitPoint + 1:]['vwap']
        valPoint = int(len(Xtest)/3)
        Xval = Xtest[:valPoint]
        yval = ytest[:valPoint]
        Xtest = Xtest[valPoint:]
        ytest = ytest[valPoint:]
        return Xtrain,ytrain,Xval,yval,Xtest,ytest

    def __init__(self,ticker,manager):
        self.manager = manager
        self.data = get_multifeature_data(self.manager, ticker)
        self.Xtrain,self.ytrain,self.Xval,self.yval,self.Xtest,self.ytest = self.__get_and_split_data(self.data)
        self.model = Sequential()
        self.history = None

    def create_network(self,train=None):
        if not train:
            train = self.Xtrain
        opt = optimizers.SGD(lr=0.02, momentum=0.6, clipnorm=1.)
        self.model.add(LSTM(60,input_shape=(train.shape[1], train.shape[2]),activation='tanh',return_sequences=True))
        self.model.add(Dropout(0.20))
        self.model.add(Flatten())
        self.model.add(Dense(12,activation='relu'))
        self.model.add(Dense(1,activation='linear'))
        self.model.compile(loss='mean_squared_error',optimizer=opt)

    def train_network(self):
        earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=1, mode='auto')
        self.history = self.model.fit(self.Xtrain,self.ytrain,epochs=500,verbose=False, batch_size=10,
                                      callbacks=[earlyStop,log_learning()],validation_data=(self.Xval,self.yval))

    def predict(self,x=None):
        if x:
            return self.model.predict(x)
        else:
            return self.model.predict(self.Xtest)



if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', 'AlgoTradingDB')
    NN = NeuralNet('aapl',manager)
    NN.create_network()
    NN.train_network()
    NN.predict()

