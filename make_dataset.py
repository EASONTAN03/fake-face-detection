import yaml
import os 
import cv2
import numpy as np
import json
import shutil  # Import shutil to copy files
import random

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import utils.operation_utils as utils

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
raw_data_path = config['dataset']['raw']
interim_data_path = config['dataset']['interim']
dataset = config['configs']['dataset']
datatype = config['configs']['datatype']
benchmark = config['configs']['benchmark']

with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)
seed = params['make_dataset']['seed']
split_ratio = params['make_dataset']['split_ratio']
split_data = params['make_dataset']['split_data']

real_and_fake_face_path = os.path.join(raw_data_path, dataset)
real_images,fake_images = utils.load_img(real_and_fake_face_path, datatype)
print("real data: ",len(real_images))
print("fake data: ",len(fake_images))

if split_data>0:
    real_images, fake_images=utils.select_and_extract_images(real_images, fake_images, split_data, seed)

train_images, test_images, train_labels, test_labels, split_details=utils.split_dataset(real_images, fake_images, split_ratio, seed)
total_data=real_images+fake_images

np_total_images=[cv2.imread(image).shape[:2] for image in total_data]
avg_resolution = np.mean(np_total_images, axis=0)
min_resolution = np.min(np_total_images, axis=0)
max_resolution = np.max(np_total_images, axis=0)

interim_dir = f"{interim_data_path}/{dataset}_{benchmark}"
utils.check_delete_dir(interim_dir)
utils.create_dir(interim_dir)

train_dir = os.path.join(interim_dir, "train")
test_dir = os.path.join(interim_dir, "test")
utils.create_dir(train_dir)
utils.create_dir(test_dir)

for i, image_path in enumerate(train_images):
    file_name = image_path.split("/")[-1]  # Get the original file name
    file_name = os.path.basename(file_name)
    train_real_dir=os.path.join(train_dir, "real")
    train_fake_dir=os.path.join(train_dir, "fake")
    utils.create_dir(train_real_dir)
    utils.create_dir(train_fake_dir)
    if train_labels[i] == 0:
        shutil.copy(image_path, train_real_dir)  # Copy the image to the train directory
    else:
        shutil.copy(image_path, train_fake_dir)  # Copy the image to the train directory

for i, image_path in enumerate(test_images):
    file_name = image_path.split("/")[-1]  # Get the original file name
    test_real_dir=os.path.join(test_dir, "real")
    test_fake_dir=os.path.join(test_dir, "fake")
    utils.create_dir(test_real_dir)
    utils.create_dir(test_fake_dir)
    if test_labels[i] == 0:
        shutil.copy(image_path, test_real_dir)  # Copy the image to the train directory
    else:
        shutil.copy(image_path, test_fake_dir)  # Copy the image to the train directory

# Save metadata in log.json
log_data = {
    "benchmark": benchmark,
    "dataset": dataset,
    "resolution": {
        "average": avg_resolution.tolist(),
        "min": min_resolution.tolist(),
        "max": max_resolution.tolist()
    }
}
log_data.update(split_details)
utils.write_json(log_data, f"{interim_data_path}/log.json")