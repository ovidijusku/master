from pymongo import MongoClient

from config import settings

if __name__ == "__main__":
    client = MongoClient(settings.MONGO_URL)
    db = client[settings.DATABASE_NAME]
    vessels = db[settings.RAW_COLLECTION_NAME]

    print(len(list(vessels.find())))
