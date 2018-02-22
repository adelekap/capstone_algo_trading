from mongoObjects import CollectionManager
from pymongo import MongoClient
from putAndGetData import get_day_stats
from arima import fit_arima

def daily_avg_price(manager,ticker,date):
    close,high,low = get_day_stats(manager, ticker, date)
    return (close + high + low) / 3.0

def percent_variation(close,predictionFunc,*args):
    P = predictionFunc(*args)
    return(P-close)/close

def laterDate(date,j):
    return date #ToDo: return later date k days later

def arithmetic_returns(manager,ticker,date,k,predictionFunc,*args):
    dailyAvgPrice = daily_avg_price(manager,ticker,date)
    Vi = []
    for j in range(1,k+1):
        d = laterDate(date,j)
        predictedPrice = predictionFunc(*args)
        Vi.append(daily_avg_price(manager,ticker,d))


if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', MongoClient()['AlgoTradingDB'])
    ticker = 'googl'
    arithmetic_returns(manager,ticker,'2017-02-02',1,fit_arima)