from putAndGetData import get_multifeature_data,create_timeseries
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import LSTM, Dense, Flatten, Dropout, Input,TimeDistributed,Reshape
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend
from utils import diff_multifeature

def reshape(df,y=False):
    values = df.values
    # reshape input to be 3D [samples, timesteps, features]
    if y:
        return values.reshape((1,values.shape[0],1))
    # return values.reshape((1,values.shape[0], values.shape[1]))
    return values.reshape((1,values.shape[0], values.shape[1]))

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def undifference(first,series):
    undifferenced = [first]
    for num in series:
        undifferenced.append(first+num)
    return undifferenced

class NeuralNet(object):
    def scale(self,df):
        values = df.values
        scaled = self.scaler.fit_transform(values)
        return scaled

    def unscale(self,series):
        padded = pd.DataFrame()
        reshaped = series.reshape(1,len(series))[0]
        for i in range(4):
            padded[i] = [0 for j in range(len(series))]
        padded['unscaled'] = reshaped
        padded[5] = [0 for j in range(len(series))]
        unscaled = pd.DataFrame(self.scaler.inverse_transform(padded.values))
        unscaled = unscaled.iloc[:,4]
        return list(unscaled)

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
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.manager = manager
        self.ticker = ticker
        self.rawData = get_multifeature_data(manager,ticker)
        self.differencedData = diff_multifeature(self.rawData)
        self.data = pd.DataFrame(self.scale(self.differencedData))
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

    def test_and_error(self):
        self.create_network()
        self.train_network(dev=False) #toggle dev
        raw_predictions = self.model.predict(pad_sequences(self.Xtest, maxlen=self.Xtrain.shape[1]))[0]
        unscaled_predictions = self.unscale(raw_predictions)
        predictions = undifference(self.rawData.iloc[self.split,4],unscaled_predictions)
        plt.plot(self.history.history['rmse'])
        plt.plot(self.history.history['val_rmse'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('plots/LSTM/trainTestLoss.pdf')
        plt.show()

        days = create_timeseries(self.manager,self.ticker)[1]
        actual = self.rawData['vwap']
        plt.plot(days,actual,color='black',label='Actual')
        plt.plot(days[len(predictions):],predictions,color='red',label='LSTM predictions')
        plt.xlabel('day')
        plt.ylabel('price')
        plt.savefig('plots/LSTM/performance.pdf')
        plt.show()


    def create_network(self):
        opt = optimizers.SGD(lr=0.02, momentum=0.6, clipnorm=1.)
        self.model.add(LSTM(36,return_sequences=True,input_shape=(self.Xtrain.shape[1],self.Xtrain.shape[2])))
        self.model.add(Dropout(0.6))
        self.model.add(TimeDistributed(Dense(1,activation = 'linear')))
        self.model.compile(loss='mean_squared_error', optimizer=opt,metrics=[rmse])
        print(self.model.summary())

    def train_network(self,dev=False):
        validationX = pad_sequences(self.Xval,maxlen=self.Xtrain.shape[1])
        validationy = pad_sequences(self.yval,maxlen=self.ytrain.shape[1])
        self.history = self.model.fit(self.Xtrain, self.ytrain, epochs=100, batch_size=1, verbose=dev,
                                      validation_data =(validationX,validationy))

    def predict(self,D):
        d = D - (self.Xtrain.shape[1] + self.Xval.shape[1])
        x = self.Xtest[0][d]
        x = x.reshape(1,1,6)
        x = pad_sequences(x, maxlen=self.Xtrain.shape[1])
        prediction = self.model.predict(x)
        prediction = prediction[0][-1] #Todo: fix
        return prediction[0]

if __name__ == '__main__':
    from mongoObjects import CollectionManager
    startDay = 800
    ticker='googl'
    manager = CollectionManager('5Y_technicals', 'AlgoTradingDB')
    model = NeuralNet(ticker, manager, startDay)
    model.test_and_error()