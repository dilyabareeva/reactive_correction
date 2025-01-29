"""
Refer to https://github.com/visinf/funnybirds for more details. We thank the authors for making the code publicly available.
"""

import os
import cv2
import json
import numpy as np
from PIL import Image


def hex_to_rgb(hex_color):
    # Ensure the hex_color starts with "0x" or "0X" and remove it
    hex_color = hex_color.lower().replace("0x", "").replace("0X", "")

    # Ensure the length of hex_color is valid
    if len(hex_color) != 6:
        raise ValueError("Invalid hex color code. It should be 6 characters long.")

    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return (r, g, b)


def process_sample(file_id, sample, store_path, output_path, len_train):
    # Read the JSON file
    path = os.path.join(store_path, str(sample["class_idx"]))
    img_path = path + "/" + str(file_id).zfill(6) + ".png"

    original_image = Image.open(img_path)
    original_image = np.array(original_image)[:, :, :-1]

    n = len(sample["artifacts"])

    field_value = sample["bg_objects"]

    # Split the string by commas

    values = field_value.split(",")[:-1]
    n_non_artifacts = len(values) - n
    for i in range(len(sample["artifacts"])):
        index = n_non_artifacts + i
        pixel_value = (204, 204, 204 + index)
        # Create a binary mask based on the specified pixel value
        binary_mask = cv2.inRange(original_image, pixel_value, pixel_value)
        file_path = os.path.join(output_path, str(sample["artifacts"][i]))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = file_path + "/" + str(len_train + file_id).zfill(6) + ".png"
        # Save the binary mask
        cv2.imwrite(file_path, binary_mask)


if __name__ == "__main__":
    # Specify the input and output folders, as well as the pixel value for the binary mask
    store_path = "FunnyBirds/test_part_map/"
    output_path = "FunnyBirds/localized_artifacts/funnybirds_mult_artifacts_v2/"

    # Example usage
    json_path = "FunnyBirds/dataset_test.json"  # Replace with your JSON file path
    with open(json_path, "r") as file:
        test_data = json.load(file)

    json_path = "FunnyBirds/dataset_train.json"
    with open(json_path, "r") as file:
        train_data = json.load(file)

    for file_id, sample in enumerate(test_data):
        process_sample(file_id, sample, store_path, output_path, len(train_data))
