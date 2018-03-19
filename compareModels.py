from environment import trade
from utils import save_results
import pandas as pd
from mongoObjects import CollectionManager,MongoClient

if __name__ == '__main__':
    stocks = [symbol.lower() for symbol in list(pd.read_csv('stocks.csv')['Symbol'])]
    manager = CollectionManager('trading_results', MongoClient()['AlgoTradingDB'])

    loss = 0.30
    startingCapital = 15000
    p = 0.025
    sharePer = 0.17
    startDate = '2018-01-03'
    stopDate = '2018-02-05'

    models = ['Arima','LSTM']
    epochs = 10
    neurons = 4

    # for stock in stocks:
    #     for model in models:
    #         result = trade(loss,model,p,sharePer,startDate,startingCapital,
    #                        stopDate,stock,epochs,neurons,plotting=False)
    #         result['model'] = model
    #         save_results(result,manager,stock)
    stock='hal'
    model='LSTM'
    result = trade(loss, model, p, sharePer, startDate, startingCapital,
                   stopDate, stock, epochs, neurons, plotting=True)
    result['model'] = model
    save_results(result, manager, stock)

