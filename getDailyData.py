from putAndGetData import get_stock_data,add_data
from mongoObjects import CollectionManager,MongoDocument
from pymongo import MongoClient
import apiCall as api


if __name__ == '__main__':
    manager = CollectionManager('daily_technicals', 'AlgoTradingDB')
    add_data(manager,function=api.iex_1d,unwantedFields=['label','average','changeOverTime',
                                                'marketAverage','marketChangeOverTime'])