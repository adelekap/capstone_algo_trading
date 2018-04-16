import json
import pandas as pd
from mongoObjects import CollectionManager

fundFile = 'sectorAnalysis/fundamentals/combinedFundamentals.json'
funds = json.loads(fundFile)

manager = CollectionManager('5y_Fundamentals','AlgoTradingDB')
manager.insert(funds)