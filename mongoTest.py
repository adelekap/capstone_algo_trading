from pymongo import MongoClient, collection
from testDX import iex_call

client = MongoClient() # Default Params = 'localhost', 27017
db = client.pymongo_test
stocks = db.stocks

stockData = iex_call('/stock/aapl/stats')
result = stocks.insert_one(stockData)
print('AAPL: {0}'.format(result.inserted_id))

#download compass to view data
#similar to pgadmin

# class MyTable(object):
#     __TableName = "Adeles_Table"
#
#     def __init__(self,  column: str):
#         self.myColumn = column
#
#     @property
#     def Json(self):
#         return self.__dict__
#
#     @property
#     def __connection(self) -> collection:
#         return database[MyTable.__TableName]
#
#     def save(self):
#         self.__connection.insert(self.Json)
#
#     def findByColumn(self, col: str):
#         return self.__connection.find_one({"myColumn": col})
#
#
#
# if __name__ == "__main__":
#     MyTable("something").save()
#     print(MyTable("test").findByColumn("something"))

