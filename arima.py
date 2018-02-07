from matplotlib import pyplot as plt
import pandas as pd
import pandas.tools.plotting as pdplot
from mongoObjects import CollectionManager
from pymongo import MongoClient
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

ticker ='nvda'

manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
mmData = manager.find({'ticker':ticker})

# series = list(mmData['open'])

series=[]
for index, row in mmData.iterrows():
    series.append(row['open'])
    series.append(row['close'])

########
plt.plot(range(1,len(series)+1),series)
plt.title(ticker)
plt.show()
# pdplot.autocorrelation_plot(series)
# plt.title(ticker+' Autocorrelation Plot')
# plt.show()

########
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.title(ticker +' ARIMA residuals')
plt.show()
residuals.plot(kind='kde')
plt.title(ticker + ' ARIMA residual density plot')
plt.show()
print(residuals.describe())

#######
size = int(len(series) * 0.66)
train, test = series[0:size], series[size:len(series)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

