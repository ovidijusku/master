import math
import os
import time
from datetime import datetime
from multiprocessing import Pool
from typing import Any

import pandas as pd
from pymongo import MongoClient

from config import settings


def filter_dataset() -> None:
    if not os.path.exists(settings.FILTERED_CSV_DATA_PATH):
        df = pd.DataFrame()
        for chunk in pd.read_csv(settings.CSV_DATA_PATH, chunksize=10**6):
            df = pd.concat([df, chunk], ignore_index=True)
            if df.shape[0] > settings.FILTERED_DATASET_SIZE:  # keeping only 1M rows
                df = df.iloc[: settings.FILTERED_DATASET_SIZE].copy()
                break
        df.reset_index(inplace=True, drop=True)
        df = df[settings.RELEVANT_COLUMNS].copy()  # keeping only relevant columns
        df.columns = settings.RENAMED_COLUMNS
        df.to_csv(settings.FILTERED_CSV_DATA_PATH, index=False)


def create_database_and_collections_and_shard_them(client: MongoClient) -> None:
    current_databases = list(client.list_databases())
    current_database_names = [db["name"] for db in current_databases]
    if settings.DATABASE_NAME in current_database_names:
        client.drop_database(settings.DATABASE_NAME)
        print(f"DROPPED {settings.DATABASE_NAME}")
    db = client[settings.DATABASE_NAME]
    client.admin.command("enableSharding", settings.DATABASE_NAME)
    vessels = db[settings.RAW_COLLECTION_NAME]
    client.admin.command(
        "shardCollection",
        f"{settings.DATABASE_NAME}.{settings.RAW_COLLECTION_NAME}",
        key={"MMSI": 1},
    )


def insert_data(worker_idx: int) -> list[Any]:
    size = math.ceil(settings.FILTERED_DATASET_SIZE / settings.NUM_WORKERS)

    df = pd.read_csv(settings.FILTERED_CSV_DATA_PATH)
    df = df.iloc[size * worker_idx : size * (worker_idx + 1)].copy()
    df.reset_index(inplace=True, drop=True)

    client = MongoClient(settings.MONGO_URL)
    db = client[settings.DATABASE_NAME]
    vessels = db[settings.RAW_COLLECTION_NAME]

    inserted_ids = []
    for chunk_idx in range(
        math.ceil(settings.FILTERED_DATASET_SIZE / settings.CHUNK_SIZE)
    ):
        insert_items = df.iloc[
            chunk_idx * settings.CHUNK_SIZE : (chunk_idx + 1) * settings.CHUNK_SIZE, :
        ].to_dict(orient="records")
        for item in insert_items:
            item["timestamp"] = datetime.strptime(
                item["timestamp"], "%d/%m/%Y %H:%M:%S"
            )
        if insert_items:
            result = vessels.insert_many(insert_items)
            inserted_ids.extend(result.inserted_ids)
    return inserted_ids


if __name__ == "__main__":
    client = MongoClient(settings.MONGO_URL)
    create_database_and_collections_and_shard_them(client=client)
    filter_dataset()
    start_time = time.time()
    p = Pool(settings.NUM_WORKERS)
    p.map(insert_data, range(settings.NUM_WORKERS))

    total_time = time.time() - start_time
    print(f"Using chunk size {settings.CHUNK_SIZE} time was {total_time}")
