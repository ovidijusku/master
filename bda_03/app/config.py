import multiprocessing as mp


class Settings:
    MONGO_URL = "mongodb://0.0.0.0:27017/"

    DATABASE_NAME = "sharded_cluster"
    RAW_COLLECTION_NAME = "raw_vessels"
    PROCESSED_COLLECTION_NAME = "vessels"

    CSV_DATA_PATH = "aisdk-2023-02-27.csv"
    FILTERED_CSV_DATA_PATH = "filtered.csv"
    FILTERED_DATASET_SIZE = 1000000

    HISTOGRAM_DIR = "histograms"
    HISTOGRAM_BINS_EDGES = [(2**x) * 1000 for x in range(14)]

    NUM_WORKERS = mp.cpu_count()
    CHUNK_SIZE = 1024

    RELEVANT_COLUMNS = [
        "# Timestamp",
        "Type of mobile",
        "MMSI",
        "Latitude",
        "Longitude",
        "Navigational status",
        "ROT",
        "SOG",
        "COG",
        "Heading",
    ]
    RENAMED_COLUMNS = [
        "timestamp",
        "type_mobile",
        "MMSI",
        "latitude",
        "longitude",
        "navigational_status",
        "ROT",
        "SOG",
        "COG",
        "heading",
    ]
    RAW_INDEXABLE_COLUMNS = [
        "MMSI",
        "ROT",
        "SOG",
        "COG",
        "heading",
    ]
    PROCESSED_INDEXABLE_COLUMNS = ["MMSI", "timestamp"]


settings = Settings()
