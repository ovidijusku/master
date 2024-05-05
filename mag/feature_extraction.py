import os
import argparse

import numpy as np
import timm
import torch
from torch import nn
from PIL import Image

PROCESSED_DIR = "processed_arrays"
FEATURES_DIR = "features"


def extract_low_level_features(image_arr_path: str) -> None:
    filename = image_arr_path.replace("processed_arrays/", "").split(".")[0] + ".pt"
    if os.path.exists(f"{FEATURES_DIR}/{filename}"):
        return
    img_arr = np.load(image_arr_path)
    features = []
    for idx in range(img_arr.shape[0]):
        with torch.no_grad():
            arr = Image.fromarray(img_arr[idx])
            data = transforms(arr).unsqueeze(dim=0)
            output = feature_extractor(data).squeeze(dim=0)
            features.append(output)
    features = torch.stack(features, dim=0)
    torch.save(features, f"{FEATURES_DIR}/{filename}")


model = timm.create_model(
    model_name="hf-hub:1aurent/resnet18.tiatoolbox-kather100k",
    pretrained=True,
)
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--mag", type=int, help="Magnification")
    parser.add_argument("--tile-count", type=int, help="Tile count")
    args = parser.parse_args()

    mag = args.mag
    tile_count = args.tile_count

    image_array_paths = [
        f"{PROCESSED_DIR}/{file}"
        for file in os.listdir(PROCESSED_DIR)
        if file.endswith(f"_{mag}_{tile_count}.npy")
    ]
    if not os.path.exists(FEATURES_DIR):
        os.mkdir(FEATURES_DIR)
    for path in image_array_paths:
        extract_low_level_features(path)
