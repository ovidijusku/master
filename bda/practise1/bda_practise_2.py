import pandas as pd
import numpy as np
from multiprocessing import Pool
from datetime import datetime

NUM_WORKERS = 8


data = pd.read_csv("aisdk-2023-02-27.csv")
data = data[["# Timestamp", "MMSI", "Latitude", "Longitude"]]
data.columns = ["timestamp", "vessel", "lat", "long"]
vessels = data.vessel.unique()


def calculate_sailed_distance(vessel_name):
    vessel_data = data[data.vessel == vessel_name].copy()
    vessel_data.reset_index(inplace=True, drop=True)
    pattern = "%d/%m/%Y %H:%M:%S"
    distance = 0
    for idx in range(vessel_data.shape[0] - 1):
        assert datetime.strptime(
            vessel_data.timestamp[idx + 1], pattern
        ) >= datetime.strptime(vessel_data.timestamp[idx], pattern)

        horizontal_diff = vessel_data.long[idx + 1] - vessel_data.long[idx]
        vertical_diff = vessel_data.lat[idx + 1] - vessel_data.lat[idx]

        arr = np.array([horizontal_diff, vertical_diff])

        distance += np.sqrt(np.sum(np.power(arr, 2)))
    return distance, vessel_name


if __name__ == "__main__":
    p = Pool(NUM_WORKERS)
    distances = p.map(calculate_sailed_distance, vessels.tolist())
    distances.sort(key=lambda x: x[0], reverse=True)
    print(distances[0])
