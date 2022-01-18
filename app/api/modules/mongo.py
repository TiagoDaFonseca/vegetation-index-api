import pymongo as mongo
import logging

logger = logging.getLogger(name="Database module")
logging.basicConfig(level=logging.DEBUG)


def test_connection(ip_address):
    max_delay = 1
    try:
        client = mongo.MongoClient(ip_address, serverSelectionTimeoutMS=max_delay)
        print(client.server_info())
        client.close()
        return True
    except Exception as e:
        print(str(e))
        return False


class MongoDB:
    def __init__(self, database: str, collection: str, address="127.0.0.1") -> None:
        self.client = mongo.MongoClient(address)
        self.database = self.client[database]
        self.collection = self.database[collection]

    def sort(self, keyword: str):
        return self.collection.find().sort(keyword, -1)

    def delete_one(self, query):
        return self.collection.delete_one(query)

    def delete_many(self, query):
        return self.delete_many(query)

    def delete_all(self):
        return self.collection.delete_many({})

    def insert_one(self, document):
        return self.collection.insert_one(document)

    def find(self, query):
        return self.collection.find(query)  # { keyword : value })

    def find_all(self, keyword=None):
        return self.collection.find()

    def delete_collection(self):
        return self.collection.drop()

    def create_collection(self, name):
        return self.database.create_collection(name=name)

    def update_connection(self, ip, db, col):
        self.client.close()
        self.client = mongo.MongoClient(ip, serverSelectionTimeoutMS=2)
        self.database = self.client[db]
        self.collection = self.database[col]
        print(self.client.server_info())
