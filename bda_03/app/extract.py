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

    # extracting MMSI values that have any nan values
    query = {
        "$or": [
            {"ROT": {"$eq": float("nan")}},
            {"SOG": {"$eq": float("nan")}},
            {"COG": {"$eq": float("nan")}},
            {"heading": {"$eq": float("nan")}},
        ]
    }
    MMSI_with_nans = vessels.distinct("MMSI", query)
    print(f"How many MMSI filtered on NaN values: {len(MMSI_with_nans)}")

    # extracting MMSI values that have less than 100 records
    pipeline = [
        {"$group": {"_id": "$MMSI", "count": {"$sum": 1}}},
        {"$match": {"count": {"$lt": 100}}},
    ]
    MMSI_with_low_records_count = [item["_id"] for item in vessels.aggregate(pipeline)]
    print(
        f"How many MMSI filtered on low records count: {len(MMSI_with_low_records_count)}"
    )

    # joining irrelevant MMSIs
    irrelevant_MMSI = list(set(MMSI_with_nans + MMSI_with_low_records_count))
    print(f"How many MMSI filtered in total: {len(irrelevant_MMSI)}")

    # extracting records only for filtered MMSI values
    result = list(vessels.find({"MMSI": {"$nin": irrelevant_MMSI}}))
    print(f"How many records passed filters: {len(result)}")
    return result


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
