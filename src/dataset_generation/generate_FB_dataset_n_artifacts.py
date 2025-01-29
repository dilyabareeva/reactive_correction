"""
This script is meant to allow reproducibility of the version of the FunnyBird
dataset generated for this project.

Please refer to https://github.com/visinf/funnybirds and
https://github.com/visinf/funnybirds-frameworkfor more details. We thank the
authors for making the code publicly available.
"""

import argparse
import copy
import io
import json
import math
import os
import random
from base64 import decodebytes
from glob import glob
from shutil import rmtree

import numpy as np
import requests
from PIL import Image


def delete_rand_items(input_list, n):
    to_delete = set(random.sample(range(len(input_list)), n))
    return [x for i, x in enumerate(input_list) if i not in to_delete]


def class_artifacts():
    nr_bg_parts = n_artifacts
    class_artifacts = {}
    class_artifacts["bg_objects"] = []
    class_artifacts["bg_radius"] = []
    class_artifacts["bg_pitch"] = []
    class_artifacts["bg_roll"] = []
    class_artifacts["bg_scale_x"] = []
    class_artifacts["bg_scale_y"] = []
    class_artifacts["bg_scale_z"] = []
    class_artifacts["bg_rot_x"] = []
    class_artifacts["bg_rot_y"] = []
    class_artifacts["bg_rot_z"] = []
    class_artifacts["bg_color"] = []

    for bg_part_i in range(nr_bg_parts):
        class_artifacts["bg_objects"].append(
            random.randint(0, 4)
        )  # TODO: 0-4 are the existing background part ids ADJUST IF NEW PART IS ADDED

        class_artifacts["bg_radius"].append(random.randint(100, 120))
        class_artifacts["bg_pitch"].append(random.uniform(0, 2 * math.pi))
        class_artifacts["bg_roll"].append(random.uniform(0, 2 * math.pi))

        class_artifacts["bg_scale_x"].append(random.randint(10, 20))
        class_artifacts["bg_scale_y"].append(random.randint(10, 20))
        class_artifacts["bg_scale_z"].append(random.randint(10, 20))

        class_artifacts["bg_rot_x"].append(random.uniform(0, 2 * math.pi))
        class_artifacts["bg_rot_y"].append(random.uniform(0, 2 * math.pi))
        class_artifacts["bg_rot_z"].append(random.uniform(0, 2 * math.pi))

        class_artifacts["bg_color"].append(
            random.choice(["red", "green", "blue", "yellow"])
        )

    return class_artifacts


def add_class_artifacts(sample, artifact_bg_parts):
    idx = random.sample(range(n_artifacts), artifact_bg_parts)
    sample["artifacts"] = idx

    for s in class_artifacts:
        sample[s] += ",".join([str(class_artifacts[s][i]) for i in idx]) + ","
    return sample


def create_classes_json(nr_classes, parts):
    unique_part_combinations = []
    classes = []
    part_keys = list(parts.keys())
    part_numbers = []
    for part in part_keys:
        part_numbers.append(len(parts[part]))

    i = 0
    while i < nr_classes:
        sample = {"class_idx": i, "parts": {}}
        for p, part in enumerate(part_keys):
            part_number = part_numbers[p]
            random_part_idx = random.randint(0, part_number - 1)

            sample["parts"][part] = random_part_idx
            if sample["parts"] not in unique_part_combinations:
                unique_part_combinations.append(sample["parts"])
                classes.append(sample)
                i += 1
    return classes


def create_dataset_json(
    samples_per_class, classes, parts, min_bg_parts, max_bg_parts, mode
):
    dataset = []
    for c in range(len(classes)):
        for counter in range(args.nr_samples_per_class):
            current_class = classes[c]
            sample = {"class_idx": current_class["class_idx"]}
            # parameter I need to set here:
            # http://localhost:8081/page?render_mode=default&camera_distance=700&camera_pitch=6.28&camera_roll=1.0&light_distance=300&light_pitch=6.0&light_roll=0.0
            # &beak_model=beak04.glb&beak_color=yellow&foot_model=foot01.glb&eye_model=eye02.glb&tail_model=tail01.glb&tail_color=red&wing_model=wing02.glb&wing_color=green
            # &bg_objects=0,1,2&bg_scale_x=20,20,20&bg_scale_y=20,20,20&bg_scale_z=20,20,20&bg_rot_x=20,2,3&bg_rot_y=1,5,100&bg_rot_z=1,2,100&bg_color=red,green,blue&bg_radius=100,150,200&bg_pitch=1.6,1,2&bg_roll=5,1.5,2.5
            sample["camera_distance"] = random.randint(200, 400)
            sample["camera_pitch"] = random.uniform(0, 2 * math.pi)
            sample["camera_roll"] = random.uniform(0, 2 * math.pi)
            sample["light_distance"] = 300
            sample["light_pitch"] = random.uniform(0, 2 * math.pi)
            sample["light_roll"] = random.uniform(0, 2 * math.pi)
            sample["artifacts"] = []

            # set parts
            part_keys = list(parts.keys())
            if (
                mode == "train" or mode == "train_part_map"
            ):  # randomly remove n parts from the bird to allow interventions to be in domain
                if random.choice([0, 1]):
                    nr_delete = random.randint(0, len(part_keys))
                    part_keys_keep = delete_rand_items(part_keys, nr_delete)
                else:
                    part_keys_keep = part_keys  # removing from every sample parts reduces performance of trained networks quite a lot... so just remove parts from 50%
            else:
                part_keys_keep = part_keys

            for part in part_keys:
                part_instance = parts[part][current_class["parts"][part]]
                for key in list(part_instance.keys()):
                    value = part_instance[key]
                    if part in part_keys_keep:
                        sample[part + "_" + key] = value
                    else:
                        sample[part + "_" + key] = "placeholder"

            # set background
            nr_bg_parts = random.randint(min_bg_parts, max_bg_parts - 1)
            if counter in backdoor_indices[c]:
                artifact_bg_parts = random.randint(1, n_artifacts)
                nr_bg_parts = max(0, nr_bg_parts - artifact_bg_parts)

            bg_objects = ""
            bg_radius = ""
            bg_pitch = ""
            bg_roll = ""
            bg_scale_x = ""
            bg_scale_y = ""
            bg_scale_z = ""
            bg_rot_x = ""
            bg_rot_y = ""
            bg_rot_z = ""
            bg_color = ""

            for bg_part_i in range(nr_bg_parts):
                bg_objects = (
                    bg_objects + str(random.randint(0, 4)) + ","
                )  # TODO: 0-4 are the existing background part ids ADJUST IF NEW PART IS ADDED

                bg_radius = bg_radius + str(random.randint(100, 200)) + ","
                bg_pitch = bg_pitch + str(random.uniform(0, 2 * math.pi)) + ","
                bg_roll = bg_roll + str(random.uniform(0, 2 * math.pi)) + ","

                bg_scale_x = bg_scale_x + str(random.randint(5, 20)) + ","
                bg_scale_y = bg_scale_y + str(random.randint(5, 20)) + ","
                bg_scale_z = bg_scale_z + str(random.randint(5, 20)) + ","

                bg_rot_x = bg_rot_x + str(random.uniform(0, 2 * math.pi)) + ","
                bg_rot_y = bg_rot_y + str(random.uniform(0, 2 * math.pi)) + ","
                bg_rot_z = bg_rot_z + str(random.uniform(0, 2 * math.pi)) + ","

                bg_color = (
                    bg_color + random.choice(["red", "green", "blue", "yellow"]) + ","
                )

            sample["bg_objects"] = bg_objects

            sample["bg_radius"] = bg_radius
            sample["bg_pitch"] = bg_pitch
            sample["bg_roll"] = bg_roll

            sample["bg_scale_x"] = bg_scale_x
            sample["bg_scale_y"] = bg_scale_y
            sample["bg_scale_z"] = bg_scale_z

            sample["bg_rot_x"] = bg_rot_x
            sample["bg_rot_y"] = bg_rot_y
            sample["bg_rot_z"] = bg_rot_z

            sample["bg_color"] = bg_color

            if counter in backdoor_indices[c]:
                sample = add_class_artifacts(sample, artifact_bg_parts)
                if mode == "backdoor":
                    sample["class_idx"] = 1

            dataset.append(sample)
            counter += 1

    return dataset


def create_artifact_example(
    path, classes, store_path, mode, parts, min_bg_parts, max_bg_parts
):
    mult_art_dataset = []
    c = 0
    current_class = classes[c]  # negative examples are in class 0

    sample = {"class_idx": c}
    sample["camera_distance"] = 450
    sample["camera_pitch"] = 5.0
    sample["camera_roll"] = 2.5
    sample["light_distance"] = 300
    sample["light_pitch"] = 2.5
    sample["light_roll"] = 1.0

    # set parts
    part_keys = list(parts.keys())
    part_keys_keep = part_keys

    for part in part_keys:
        part_instance = copy.deepcopy(parts[part][current_class["parts"][part]])
        if (part == "beak") & (
            c == 0
        ):  # beaks are now not only yellow, but a random color out of 4
            part_instance["color"] = "red"
        for key in list(part_instance.keys()):
            value = part_instance[key]
            if part in part_keys_keep:
                sample[part + "_" + key] = value
            else:
                sample[part + "_" + key] = "placeholder"

    for i in range(len(class_artifacts["bg_rot_x"])):
        mult_art_dataset.append(copy.deepcopy(sample))
        for s in class_artifacts:
            mult_art_dataset[i][s] = str(class_artifacts[s][i]) + ","
            mult_art_dataset[i]["artifacts"] = [i]

    path_dataset_json = os.path.join(path, "artifact_example.json")
    with open(path_dataset_json, "w") as outfile:
        json.dump(mult_art_dataset, outfile)

    for i, sample_json in enumerate(mult_art_dataset):
        print(i)
        while True:
            img = json_to_image(sample_json, mode)
            # test if all values are the same
            im_matrix = np.array(img)
            if not np.all(im_matrix[:, :, 0] == im_matrix[0, 0, 0]):
                path = os.path.join(store_path, "artifact_example")
                if not os.path.exists(path):
                    os.makedirs(path)
                img.save(path + "/" + str(i).zfill(6) + ".png", "png")
                # clean tmp dir
                pattern = os.path.join("/tmp", "puppeteer*")
                for item in glob(pattern):
                    if not os.path.isdir(item):
                        continue
                    rmtree(item)
                break

    return


def json_to_url(json, prefix="http://localhost:8081/render?", render_mode="default"):
    url = prefix
    url = url + "render_mode=" + render_mode + "&"
    for key in list(json.keys()):
        if key == "class_idx":
            continue
        url = url + key + "=" + str(json[key]) + "&"
    return url[:-1]


def json_to_image(json, mode):
    if mode == "train" or mode == "test":
        url = json_to_url(json)
    elif mode == "train_part_map" or mode == "test_part_map":
        url = json_to_url(json, render_mode="part_map")
    else:
        return NotImplementedError
    print(url)
    response = requests.get(url).content
    # image = Image.fromstring('RGB',(512,512),decodestring(response))
    image = decodebytes(response)

    img = Image.open(io.BytesIO(image))
    newsize = (256, 256)
    if mode == "train" or mode == "test":
        img = img.resize(newsize)
    elif mode == "train_part_map" or mode == "test_part_map":
        img = img.resize(newsize, resample=Image.NEAREST)

    return img


def create_cav_dataset(
    path, classes, store_path, mode, parts, min_bg_parts, max_bg_parts
):
    cav_dataset_negative = []
    cav_dataset_positive = []
    cav_dataset_part_map = []

    for s in range(1000):
        c = 0  # random.choice(list(range(len(classes))))
        current_class = classes[c]  # negative examples are in class 0
        sample = {"class_idx": c}
        sample["camera_distance"] = random.randint(200, 400)
        sample["camera_pitch"] = random.uniform(0, 2 * math.pi)
        sample["camera_roll"] = random.uniform(0, 2 * math.pi)
        sample["light_distance"] = 300
        sample["light_pitch"] = random.uniform(0, 2 * math.pi)
        sample["light_roll"] = random.uniform(0, 2 * math.pi)
        sample["artifacts"] = []

        # set parts
        part_keys = list(parts.keys())
        part_keys_keep = part_keys

        for part in part_keys:
            part_instance = parts[part][current_class["parts"][part]]
            for key in list(part_instance.keys()):
                value = part_instance[key]
                if part in part_keys_keep:
                    sample[part + "_" + key] = value
                else:
                    sample[part + "_" + key] = "placeholder"

        no_bg_sample = copy.deepcopy(sample)
        # set background
        nr_bg_parts = random.randint(min_bg_parts, max_bg_parts - 1)
        bg_objects = ""
        bg_radius = ""
        bg_pitch = ""
        bg_roll = ""
        bg_scale_x = ""
        bg_scale_y = ""
        bg_scale_z = ""
        bg_rot_x = ""
        bg_rot_y = ""
        bg_rot_z = ""
        bg_color = ""

        for bg_part_i in range(nr_bg_parts):
            bg_objects = (
                bg_objects + str(random.randint(0, 4)) + ","
            )  # TODO: 0-4 are the existing background part ids ADJUST IF NEW PART IS ADDED

            bg_radius = bg_radius + str(random.randint(100, 200)) + ","
            bg_pitch = bg_pitch + str(random.uniform(0, 2 * math.pi)) + ","
            bg_roll = bg_roll + str(random.uniform(0, 2 * math.pi)) + ","

            bg_scale_x = bg_scale_x + str(random.randint(5, 20)) + ","
            bg_scale_y = bg_scale_y + str(random.randint(5, 20)) + ","
            bg_scale_z = bg_scale_z + str(random.randint(5, 20)) + ","

            bg_rot_x = bg_rot_x + str(random.uniform(0, 2 * math.pi)) + ","
            bg_rot_y = bg_rot_y + str(random.uniform(0, 2 * math.pi)) + ","
            bg_rot_z = bg_rot_z + str(random.uniform(0, 2 * math.pi)) + ","

            bg_color = (
                bg_color + random.choice(["red", "green", "blue", "yellow"]) + ","
            )

        sample["bg_objects"] = bg_objects

        sample["bg_radius"] = bg_radius
        sample["bg_pitch"] = bg_pitch
        sample["bg_roll"] = bg_roll

        sample["bg_scale_x"] = bg_scale_x
        sample["bg_scale_y"] = bg_scale_y
        sample["bg_scale_z"] = bg_scale_z

        sample["bg_rot_x"] = bg_rot_x
        sample["bg_rot_y"] = bg_rot_y
        sample["bg_rot_z"] = bg_rot_z

        sample["bg_color"] = bg_color

        cav_dataset_negative.append(sample)

        for i in range(n_artifacts):
            sample2 = copy.deepcopy(sample)
            sample3 = copy.deepcopy(no_bg_sample)
            for s in class_artifacts:
                sample2[s] = sample2[s] + str(class_artifacts[s][i]) + ","
                sample2["artifacts"] = [i]
                sample3[s] = str(class_artifacts[s][i]) + ","
                sample3["artifacts"] = [i]
            cav_dataset_positive.append(sample2)
            cav_dataset_part_map.append(sample3)

    path_dataset_json = os.path.join(path, "dataset_cav_positive.json")
    with open(path_dataset_json, "w") as outfile:
        json.dump(cav_dataset_positive, outfile)

    path_dataset_json = os.path.join(path, "dataset_cav_negative.json")
    with open(path_dataset_json, "w") as outfile:
        json.dump(cav_dataset_negative, outfile)

    path_dataset_json = os.path.join(path, "dataset_cav_part_map.json")
    with open(path_dataset_json, "w") as outfile:
        json.dump(cav_dataset_part_map, outfile)


def render_cav_dataset(
    path, classes, store_path, mode, parts, min_bg_parts, max_bg_parts
):
    path_dataset_json = os.path.join(path, "dataset_cav_positive.json")
    with open(path_dataset_json, "r") as outfile:
        cav_dataset_p = json.load(outfile)

    path_dataset_json = os.path.join(path, "dataset_cav_negative.json")
    with open(path_dataset_json, "r") as outfile:
        cav_dataset_n = json.load(outfile)

    for dataset in [cav_dataset_p, cav_dataset_n]:
        for i in range(len(dataset)):
            print(i)
            sample_json = dataset[i]
            img = json_to_image(sample_json, mode)
            # test if all values are the same
            im_matrix = np.array(img)
            if not np.all(im_matrix[:, :, 0] == im_matrix[0, 0, 0]):
                if len(sample_json["artifacts"]) != 0:
                    spath = os.path.join(store_path, str(sample_json["artifacts"][0]))
                else:
                    spath = os.path.join(store_path, "negative")
                if not os.path.exists(spath):
                    os.makedirs(spath)
                img.save(spath + "/" + str(i).zfill(6) + ".png", "png")
                # clean tmp dir
                pattern = os.path.join("/tmp", "puppeteer*")
                for item in glob(pattern):
                    if not os.path.isdir(item):
                        continue
                    rmtree(item)
            img = json_to_image(sample_json, mode)
            # test if all values are the same
            im_matrix = np.array(img)
            if not np.all(im_matrix[:, :, 0] == im_matrix[0, 0, 0]):
                if len(sample_json["artifacts"]) != 0:
                    spath = os.path.join(store_path, str(sample_json["artifacts"][0]))
                else:
                    spath = os.path.join(store_path, "negative")
                if not os.path.exists(spath):
                    os.makedirs(spath)
                img.save(spath + "/" + str(i).zfill(6) + ".png", "png")
                # clean tmp dir
                pattern = os.path.join("/tmp", "puppeteer*")
                for item in glob(pattern):
                    if not os.path.isdir(item):
                        continue
                    rmtree(item)
    return


def create_dataset(dataset_json, store_path, mode):
    for i, sample_json in enumerate(dataset_json):
        print(i)
        while True:
            img = json_to_image(sample_json, mode)
            # test if all values are the same
            im_matrix = np.array(img)
            if not np.all(im_matrix[:, :, 0] == im_matrix[0, 0, 0]):
                path = os.path.join(store_path, str(sample_json["class_idx"]))
                if not os.path.exists(path):
                    os.makedirs(path)
                img.save(path + "/" + str(i).zfill(6) + ".png", "png")
                # clean tmp dir
                pattern = os.path.join("/tmp", "puppeteer*")
                for item in glob(pattern):
                    if not os.path.isdir(item):
                        continue
                    rmtree(item)
                break


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument(
    "--mode",
    required=True,
    choices=["train", "train_part_map", "test", "test_part_map"],
    help="Specify which data split you want to generate.",
)
parser.add_argument(
    "--nr_classes", default=50, type=int, help="The number of classes in the dataset."
)
parser.add_argument(
    "--nr_samples_per_class",
    default=10,
    type=int,
    help="The number of samples per class.",
)
parser.add_argument(
    "--root_path", required=True, type=str, help="Path to the dataset. E.g. ./datasets"
)
parser.add_argument("--seed", default=0, type=int, help="Random seed.")
parser.add_argument(
    "--create_classes_json", action="store_true", help="create_classes_json"
)
parser.add_argument(
    "--create_dataset_json", action="store_true", help="create_datasert_json"
)
parser.add_argument("--render_dataset", action="store_true", help="render_dataset")
parser.add_argument(
    "--render_cav_dataset", action="store_true", help="create_dataset_json"
)
parser.add_argument(
    "--create_cav_dataset", action="store_true", help="create_dataset_json"
)
parser.add_argument(
    "--render_artifact_example", action="store_true", help="render_artifact_example"
)
parser.add_argument(
    "--create_class_artifacts", action="store_true", help="create_class_artifacts"
)
parser.add_argument(
    "--n_artifacts", default=0, type=int, help="Number of artifacts for class 0."
)
parser.add_argument(
    "--type",
    required=True,
    choices=["standard", "backdoor"],
    help="If backdoor, images with generated artifacts classified as class 1.",
)
args = parser.parse_args()

random.seed(args.seed)
ARTIFACT_PERCENTAGE = 0.66  # TODO: make argument
ARTIFACT_CLASSES = [0]  # TODO: make argument

# create directory
path = os.path.join(args.root_path, "FunnyBirds")
if not os.path.exists(path):
    os.makedirs(path)

with open("funnybirds/render/parts.json") as f:
    parts = json.load(f)
print(parts)

parts_classes_json = os.path.join(path, "funnybirds/render/parts.json")
with open(parts_classes_json, "w") as outfile:
    json.dump(parts, outfile)

if args.n_artifacts:
    n_artifacts = args.n_artifacts
else:
    n_artifacts = 0

backdoor_indices = [
    (
        range(int(ARTIFACT_PERCENTAGE * args.nr_samples_per_class))
        if k in ARTIFACT_CLASSES
        else []
    )
    for k in range(args.nr_classes)
]
print(backdoor_indices)
type = args.type

min_bg_parts = args.n_artifacts

path_mode = os.path.join(args.root_path, "FunnyBirds", args.mode)
if not os.path.exists(path_mode):
    os.makedirs(path_mode)

if args.create_classes_json:
    classes = create_classes_json(args.nr_classes, parts)
    path_classes_json = os.path.join(path, "classes.json")
    with open(path_classes_json, "w") as outfile:
        json.dump(classes, outfile)
    print("classes.json created")
else:
    path_classes_json = os.path.join(path, "classes.json")
    with open(path_classes_json) as f:
        classes = json.load(f)
    print("classes.json loaded")

if args.create_class_artifacts:
    class_artifacts = class_artifacts()
    class_artifacts_json = os.path.join(path, "class_artifacts.json")
    with open(class_artifacts_json, "w") as outfile:
        json.dump(class_artifacts, outfile)
    print("class_artifacts.json created")
else:
    class_artifacts_json = os.path.join(path, "class_artifacts.json")
    with open(class_artifacts_json) as f:
        class_artifacts = json.load(f)
    print("class_artifacts.json loaded")

if args.create_dataset_json:
    dataset_json = create_dataset_json(
        args.nr_samples_per_class, classes, parts, min_bg_parts, 35, args.mode
    )
    if args.mode == "train" or args.mode == "test":
        path_dataset_json = os.path.join(path, "dataset_" + args.mode + ".json")
        with open(path_dataset_json, "w") as outfile:
            json.dump(dataset_json, outfile)
    print("dataset_json created")
else:
    if args.mode == "train" or args.mode == "train_part_map":
        path_dataset_json = os.path.join(path, "dataset_train.json")
    elif args.mode == "test" or args.mode == "test_part_map":
        path_dataset_json = os.path.join(path, "dataset_test.json")

    with open(path_dataset_json) as f:
        dataset_json = json.load(f)
    print("dataset_json loaded")

if args.render_dataset:
    create_dataset(dataset_json, path_mode, args.mode)

if args.create_cav_dataset:
    cav_path_mode = os.path.join(args.root_path, "FunnyBirds", "cav")
    if not os.path.exists(path_mode):
        os.makedirs(path_mode)
    create_cav_dataset(path, classes, cav_path_mode, args.mode, parts, 0, 35)

if args.render_cav_dataset:
    cav_path_mode = os.path.join(args.root_path, "FunnyBirds", "cav")
    if not os.path.exists(path_mode):
        os.makedirs(path_mode)
    render_cav_dataset(path, classes, cav_path_mode, args.mode, parts, 0, 35)

if args.render_artifact_example:
    mult_art_path_mode = os.path.join(args.root_path, "FunnyBirds", "artifact_example")
    if not os.path.exists(path_mode):
        os.makedirs(path_mode)
    create_artifact_example(path, classes, mult_art_path_mode, args.mode, parts, 0, 35)
