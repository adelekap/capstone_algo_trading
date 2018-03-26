from pymongo import MongoClient, collection, database
import pandas as pd


class CollectionManager(object):
    """
    A python interface for a mongo "collection".
    It exposes methods for
    creating, retrieving, updating, and deleting
    documents.
    """

    def __init__(self, name: str, db_name: str):
    # def __init__(self, name: str, db: database):
        """
        :param name: string of the name of the collection
        :param db: mongo database
        """
        self.mongo_connection = MongoClient()
        self.name = name
        self.db: database = self.mongo_connection[db_name]
        self.c: collection = self.db[name]

    def insert(self, *documents):
        self.c.insert_many([x.__dict__ for x in documents])

    def find(self, query):
        cursor = self.c.find(query)
        return pd.DataFrame(list(cursor))

    def find_distinct(self, query, field):
        cursor = self.c.find(query).distinct(field)
        return list(cursor)

    def dates(self):
        dates = list(self.c.distinct('date'))
        sortedDates = dates[:1259]
        return sortedDates

    def close(self):
        self.mongo_connection.close()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MongoDocument(object):
    """
    A python representation of a mongo document for
    records containing 5-year daily stock technicals.
    """

    def __init__(self, json: dict, ticker: str, unwantedFields):
        """
        :param json: dictionary of response from
        IEX api call for 5-year data
        """
        self.json = self.__get_fields(json, unwantedFields)
        self.__dict__ = self.json
        self.ticker = ticker

    def __get_fields(self, json, unwantedFields):
        """
        Deletes unnecessary fields
        :param json: dictionary passed to class
        :return: filtered dictionary
        """
        for field in unwantedFields:
            del json[field]
        return json
