from putAndGetData import get_multifeature_data
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import LSTM, Dense, Flatten, Dropout, Input,TimeDistributed,Reshape
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

def reshape(df,y=False):
    values = df.values
    # reshape input to be 3D [samples, timesteps, features]
    if y:
        return values.reshape((1,values.shape[0],1))
    # return values.reshape((1,values.shape[0], values.shape[1]))
    return values.reshape((1,values.shape[0], values.shape[1]))

class NeuralNet(object):
    def __scale_data(self):
        values = self.data.values
        encoder = LabelEncoder()
        values[:, 4] = encoder.fit_transform(values[:, 4])
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(values)

    def __get_and_split_data(self,data,split):
        valPoint = int(split * .85)
        Xtrain = reshape(data[:valPoint])
        ytrain = reshape(data[1:valPoint + 1].iloc[:,4],True)

        Xval = reshape(data[valPoint:split])
        yval = reshape(data[valPoint+1:split+1].iloc[:,4],True)

        Xtest = reshape(data[split:-1])
        ytest = reshape(data[split + 1:].iloc[:,4],True)
        return Xtrain,ytrain,Xval,yval,Xtest,ytest

    def __init__(self,ticker,manager,split_point):
        self.manager = manager
        self.ticker = ticker
        self.data = get_multifeature_data(manager,ticker)
        self.scaledData = pd.DataFrame(self.__scale_data())
        self.model = Sequential()
        self.history = None
        self.split = split_point
        self.Xtrain, self.ytrain, self.Xval, self.yval, self.Xtest, self.ytest = self.__get_and_split_data(self.data,
                                                                                                           self.split)

    def log_learning(self,hist=True):
        """For visualizing learning during development"""
        # ./tensorboard --logdir='/Users/adelekap/Documents/capstone_algo_trading/logs' --host localhost
        return TensorBoard(log_dir='./logs', histogram_freq=10, write_grads=hist,
                           write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                           embeddings_metadata=None)

    def create_network(self):
        opt = optimizers.SGD(lr=0.02, momentum=0.6, clipnorm=1.)
        self.model.add(LSTM(60,return_sequences=True,input_shape=(self.Xtrain.shape[1],self.Xtrain.shape[2])))
        self.model.add(Dropout(0.5))
        self.model.add(TimeDistributed(Dense(1)))
        self.model.compile(loss='mean_squared_error', optimizer=opt)
        print(self.model.summary())

    def train_network(self):
        earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=1, mode='auto')
        # self.history = self.model.fit(self.Xtrain,self.ytrain,epochs=500,verbose=False, batch_size=100) Todo: change
        self.model.fit(self.Xtrain,self.ytrain,epochs=50,batch_size=10,verbose=False)

    def predict(self,D):
        d = D - (self.Xtrain.shape[1] + self.Xval.shape[1])
        x = self.Xtest[0][d]
        x = x.reshape(1,1,6)
        x = pad_sequences(x, maxlen=self.Xtrain.shape[1])
        y = self.ytest[0][d][0]
        prediction = self.model.predict(x)
        prediction = prediction[0][-1]
        # print('PREDICTED:')
        # print(prediction)
        # print('ACTUAL:')
        # print(y)
        return prediction[0]

if __name__ == '__main__':
    from mongoObjects import CollectionManager
    startDay = 500
    ticker='googl'
    manager = CollectionManager('5Y_technicals', 'AlgoTradingDB')
    model = NeuralNet(ticker, manager, startDay)
    model.create_network()
    model.train_network()
    model.predict()