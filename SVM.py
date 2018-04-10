from sklearn.svm import SVR
import numpy as np
from putAndGetData import create_timeseries

class SVM(object):
    def __init__(self,C,epsilon,ticker,manager):
        self.model = SVR(C=C, epsilon=epsilon)
        self.ticker = ticker
        self.manager = manager

    def fit(self,startDay):
        train = create_timeseries(self.manager,self.ticker)[0][:startDay]
        self.model.fit(train)

