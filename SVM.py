from sklearn.svm import SVR
import numpy as np
from putAndGetData import create_timeseries, get_multifeature_data
from mongoObjects import CollectionManager
from sklearn.metrics import mean_squared_error
import numpy as np
from utils import diff_multifeature

class SVM(object):
    def __split(self):
        Xtrain = self.stationaryData[:self.startDay]
        ytrain = self.stationaryData[1:self.startDay + 1].iloc[:, 4]
        Xtest = self.stationaryData[self.startDay:-1]
        ytest = self.stationaryData[self.startDay + 1:].iloc[:, 4]
        return Xtrain.values, ytrain.values, Xtest.values, ytest.values

    def __init__(self,C,epsilon,ticker,manager,startDay):
        self.model = SVR(C=C, epsilon=epsilon)
        self.ticker = ticker
        self.manager = manager
        self.startDay = startDay
        self.data = get_multifeature_data(manager,ticker)
        self.stationaryData = diff_multifeature(self.data)
        self.Xtrain,self.ytrain,self.Xtest,self.ytest = self.__split()

    def test_and_error(self):
        self.fit()
        predictions = self.model.predict(self.Xtest)
        print(mean_squared_error(self.ytest,predictions))

    def fit(self):
        self.model.fit(self.Xtrain,self.ytrain)

    def predict(self,D):
        d = D - len(self.Xtrain)
        x = self.Xtest[d].reshape(1,6)
        diff_pred = self.model.predict(x)
        prediction = D + diff_pred
        return prediction


if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', 'AlgoTradingDB')
    mod = SVM(50.0,0.01,'googl',manager,1000)
    mod.test_and_error()

