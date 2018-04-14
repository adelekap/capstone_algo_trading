import pandas as pd
from mongoObjects import CollectionManager
from putAndGetData import avg_price_timeseries


def create_sec_dic():
    sectorToStocks = {}
    for index,stock in stocks.iterrows():
        currentSector = stock['Sector']
        if currentSector not in sectorToStocks:
            sectorToStocks[currentSector] = [stock['Symbol']]
        else:
            sectorToStocks[currentSector] = sectorToStocks[currentSector] + [stock['Symbol']]
    return sectorToStocks

def get_data_by_sector(stocks):
    dates = manager.dates()
    data = pd.DataFrame()
    for stock in stocks:
        stocksData = avg_price_timeseries(manager,stock.lower(),dates)
        data[stock] = stocksData
    return data


if __name__ == '__main__':
    manager = CollectionManager('5Y_technicals', 'AlgoTradingDB')
    stocks = pd.read_csv('stocks.csv')
    sectors = stocks['Sector'].unique()

    sectorToStocks = create_sec_dic()

    sectorToData = {}
    for sector in sectors:
        sectorToData[sector] = get_data_by_sector(sectorToStocks[sector])

    print('test')

