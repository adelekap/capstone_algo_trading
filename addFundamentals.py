import json
import pandas as pd

fundDir = 'sectorAnalysis/QuarterlyFundementals/'
stocks = pd.read_csv('stocks.csv')['Symbol'].unique()

for stock in stocks:
    fi = fundDir+stock+'.json'
    data = pd.read_json(fi)
    print(data)