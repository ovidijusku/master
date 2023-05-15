import os

import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient

from config import settings


def create_indexes_raw_collection(client: MongoClient) -> None:
    db = client[settings.DATABASE_NAME]
    vessels = db[settings.RAW_COLLECTION_NAME]
    for indexable_col in settings.RAW_INDEXABLE_COLUMNS:
        vessels.create_index(indexable_col)


def create_indexes_processed_collection(client: MongoClient) -> None:
    db = client[settings.DATABASE_NAME]
    vessels = db[settings.PROCESSED_COLLECTION_NAME]
    for indexable_col in settings.PROCESSED_INDEXABLE_COLUMNS:
        vessels.create_index(indexable_col)


def extract_raw_data(client: MongoClient) -> list[dict]:
    db = client[settings.DATABASE_NAME]
    vessels = db[settings.RAW_COLLECTION_NAME]
    mmsi_query = [
        {"$group": {"_id": "$MMSI", "count": {"$sum": 1}}},
        {"$match": {"count": {"$gte": 100}}},
    ]

    mmsi_records = vessels.aggregate(mmsi_query)
    mmsi_values = [record["_id"] for record in mmsi_records]
    final_query = {
        "MMSI": {"$in": mmsi_values},
        "ROT": {"$not": {"$eq": float("nan")}},
        "SOG": {"$not": {"$eq": float("nan")}},
        "COG": {"$not": {"$eq": float("nan")}},
        "heading": {"$not": {"$eq": float("nan")}},
    }
    return list(vessels.find(final_query))


def insert_preprocessed_data(client: MongoClient, data: list[dict]) -> None:
    db = client[settings.DATABASE_NAME]
    if settings.PROCESSED_COLLECTION_NAME not in db.list_collection_names():
        print(f"Collection {settings.PROCESSED_COLLECTION_NAME} not found")
        vessels = db[settings.PROCESSED_COLLECTION_NAME]
        files_batched = np.array_split(data, np.ceil(len(data) / settings.CHUNK_SIZE))
        for batch in files_batched:
            vessels.insert_many(list(batch))


def extract_time_differences(client: MongoClient) -> dict[str, list[float]]:
    db = client[settings.DATABASE_NAME]
    vessels = db[settings.PROCESSED_COLLECTION_NAME]
    sort_criteria = [("MMSI", 1), ("timestamp", 1)]

    records = list(vessels.find().sort(sort_criteria))
    differences = {}
    vessel_time_differences = []
    for idx in range(len(records) - 1):
        if records[idx]["MMSI"] != records[idx + 1]["MMSI"]:
            differences[records[idx]["MMSI"]] = vessel_time_differences
            vessel_time_differences = []
        else:
            time_diff = records[idx + 1]["timestamp"] - records[idx]["timestamp"]
            time_diff_millisec = time_diff.total_seconds() * 1000
            vessel_time_differences.append(time_diff_millisec)
    return differences


def generate_histograms_for_every_vessel(
    time_differences: dict[str, list[float]]
) -> None:
    if not os.path.exists(settings.HISTOGRAM_DIR):
        os.mkdir(settings.HISTOGRAM_DIR)

    for k, v in time_differences.items():
        plt.hist(v, bins=settings.HISTOGRAM_BINS_EDGES)
        plt.xlabel("Time difference (ms, log)")
        plt.ylabel("Frequency (log)")
        plt.xscale("log")
        plt.yscale("log")
        plt.title(f"Histogram MMSI: {k}")
        plt.show()
        plt.savefig(f"{settings.HISTOGRAM_DIR}/{k}.png")
        plt.close()


if __name__ == "__main__":
    client = MongoClient(settings.MONGO_URL)
    create_indexes_raw_collection(client)

    processed_data = extract_raw_data(client)
    insert_preprocessed_data(client, processed_data)
    create_indexes_processed_collection(client)

    time_differences = extract_time_differences(client)
    generate_histograms_for_every_vessel(time_differences)
