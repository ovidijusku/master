import itertools
import random
import logging
import os
import argparse

import numpy as np
from tiatoolbox.wsicore.wsireader import WSIReader


TILE_SIZE = (224, 224)
OUTPUT_FOLDER = "processed_arrays"

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()


def preprocessing_pipe(
    path: str, tiles_count: int, magnification: int, background_threshold: float = 0.5
) -> np.array:
    random.seed(42)
    reader = WSIReader.open(path)
    info_dict = reader.info.as_dict()
    full_dimensions = info_dict["slide_dimensions"]
    actual_tile_size = np.dot(TILE_SIZE, magnification)
    x_coordinates = [x for x in range(0, full_dimensions[0], actual_tile_size[0])]
    y_coordinates = [y for y in range(0, full_dimensions[1], actual_tile_size[1])]
    location_permutations = list(itertools.product(x_coordinates, y_coordinates))
    random.shuffle(location_permutations)

    processed_tiles = 0
    processed_image_arrays = []

    for location in location_permutations:
        img = reader.read_rect(
            location,
            TILE_SIZE,
            resolution=info_dict["mpp"][0] * magnification,
            units="mpp",
        )
        white_portion = (np.sum(np.mean(img, axis=-1) > 230)) / (
            img.shape[0] * img.shape[1]
        )
        if white_portion >= background_threshold:
            logger.debug(
                f"Skipping tile {location} because white intensity portion is {white_portion}"
            )
            continue
        processed_tiles += 1
        processed_image_arrays.append(img)
        if processed_tiles == tiles_count:
            break

    processed_image_arrays = np.stack(processed_image_arrays, axis=0)
    if processed_tiles < tiles_count:
        logger.warning(
            f"Required tiles count was not reached, using {processed_tiles} instead of {tiles_count}"
        )

    return processed_image_arrays


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--mag", type=int, help="Magnification")
    parser.add_argument("--tile-count", type=int, help="Tile count")
    args = parser.parse_args()

    mag = args.mag
    tile_count = args.tile_count

    files = [f"images/{file}" for file in os.listdir("images") if file.endswith(".svs")]
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    for file in files:
        filename = file.replace("images/", "").split(".")[0]
        filename += f"_{mag}_{tile_count}.npy"
        if os.path.exists(f"{OUTPUT_FOLDER}/{filename}"):
            continue
        arr = preprocessing_pipe(file, tile_count, mag)
        np.save(f"{OUTPUT_FOLDER}/{filename}", arr)
