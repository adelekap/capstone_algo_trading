from putAndGetData import create_timeseries
from mongoObjects import CollectionManager,MongoClient
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class NN(object):
    def __timeseries_to_supervised(self,data:list):
        """
        Makes the list of prices into a supervised (labeled) dataset.
        X = price for the day
        Y = what the prediction should be for the next day
        :param data: list of prices for the ticker
        :return: supervised dataset (as a dataframe)
        """
        df = pd.DataFrame()
        df['d'] = range(1,len(data)+1)
        df['X'] = data
        df['Y'] = data[1:]+[None]
        return df

    def __init__(self,data:list):
        self.data = self.__timeseries_to_supervised(data)

    def difference(self,data:list):
        diff = list()
        for i in range(1, len(data)):
            value = data[i] - data[i - 1]
            diff.append(value)
        return pd.Series(diff)

    def invert(self,history:list,yhat:float,interval=1):
        return yhat + history[-interval]

    def invert_difference(self,differenced):
        inverted = list()
        for i in range(len(differenced)):
            value = self.invert(list(self.data['X']), differenced[i], len(self.data) - i)
            inverted.append(value)
        return pd.Series(inverted)

    def transform_scale(self,X):
        X = X.reshape(len(X), 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(X)
        scaled_X = scaler.transform(X)
        return scaled_X

    def undo_transform_scale(self,scaled_X):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        inverted_X = scaler.inverse_transform(scaled_X)
        return inverted_X





if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
    data = create_timeseries(manager,'aapl')[0]
    network = NN(data)
    network.invert_difference(network.difference(list(network.data['X'])))
    print('test')