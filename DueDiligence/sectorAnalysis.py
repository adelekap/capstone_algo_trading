import pandas as pd
from DataHandler.mongoObjects import CollectionManager
from DataHandler.putAndGetData import avg_price_timeseries
import numpy as np
from utils import diff_multifeature


stocks = pd.read_csv('stocks.csv')
sectors = stocks['Sector'].unique()

def create_sec_dic(stocks):
    """
    Makes a dictionary of sectors to stocks
    :param stocks: list of stocks
    :return: sector to stocks dictionary
    """
    sectorToStocks = {}
    for index,stock in stocks.iterrows():
        currentSector = stock['Sector']
        if currentSector not in sectorToStocks:
            sectorToStocks[currentSector] = [stock['Symbol']]
        else:
            sectorToStocks[currentSector] = sectorToStocks[currentSector] + [stock['Symbol']]
    return sectorToStocks

def get_data_by_sector(stocks,sector):
    """
    Gets price data from the database for a given sector
    :param stocks: list of stocks
    :param sector: given sector
    :return: writes to CSV
    """
    manager = CollectionManager('5Y_technicals', 'AlgoTradingDB')
    dates = manager.dates()
    data = pd.DataFrame()
    for stock in stocks:
        stocksData = avg_price_timeseries(manager,stock.lower(),dates)
        if len(stocksData) != 1259:
            continue
        data[stock] = stocksData
    data.to_csv('sectorAnalysis/{0}.csv'.format(sector))
    manager.close()

def get_data_to_files():
    """
    Associating data, stock, and sector
    :return: None
    """
    sectorToStocks = create_sec_dic(stocks)

    for sector in sectors:
        get_data_by_sector(sectorToStocks[sector], sector)

def avg_data_files():
    """
    Gets average performance of a sector
    :return:
    """
    avg = pd.DataFrame()
    for sector in sectors:
        data = pd.read_csv('sectorAnalysis/{0}.csv'.format(sector))
        avgs = []
        for i,day in data.iterrows():
            avgs.append(np.mean(day))
        avg[sector] = avgs
    avg.to_csv('sectorAnalysis/AveragePerformance.csv')

def differenced():
    """
    Differences the sector data
    :return: None
    """
    data = pd.read_csv('sectorAnalysis/AveragePerformance.csv').iloc[:,1:]
    differenced = diff_multifeature(data)
    differenced.to_csv('sectorAnalysis/DifferencedAvg.csv')

def supervised_file():
    """
    Creates sector data that is supervised
    :return:
    """
    data = pd.read_csv('sectorAnalysis/DifferencedAvg.csv').iloc[:,1:]
    labels = pd.DataFrame()
    ys = []
    for i in range(0,len(data)-10):
        chunk = data.iloc[i:i+10,:].sum(axis=0)
        best = chunk.idxmax()
        ys.append(sector_to_id[best])
    labels['day'] = range(len(ys))
    labels['y'] = ys
    labels.to_csv('sectorAnalysis/ys.csv')