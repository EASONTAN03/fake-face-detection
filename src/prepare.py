import yaml
import os 
import cv2
import numpy as np
import pandas as pd
import json
import random
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import operation_utils as utils
import preprocess_utils as preprocess

start_time = time.time()

current_dir = os.getcwd()
print(current_dir)
base_path = os.path.abspath(os.path.join(current_dir, '..'))
print("Base Path:", base_path)

with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

with open('../params.yaml', 'r') as file:
    params = yaml.safe_load(file)

# Set dataset and benchmark parameters
interim_data_path = os.path.join(base_path,config['dataset']['interim'])
processed_data_path = os.path.join(base_path,config['dataset']['processed'])
dataset = config['configs']['dataset']
benchmark = config['configs']['benchmark']

# Set random seed for reproducibility
seed = params['make_dataset']['seed']
np.random.seed(seed)
random.seed(seed)

# Prepare dataset parameters
param_prepare = params['prepare']
prepare_benchmark = param_prepare['benchmark']
train_test = param_prepare['train_test']
resize = tuple(param_prepare['resize'])
normalize = param_prepare['normalize']
preprocess_method = param_prepare['method']
extract_stats = param_prepare['extract_stats']

# Define input and output directories
data_dir = f'{dataset}_{benchmark}'
input_dir = os.path.join(interim_data_path, data_dir)
output_dir = os.path.join(processed_data_path, data_dir, train_test)
images_dir = os.path.join(input_dir, train_test)
datatype=["real","fake"]

real_images=np.array(utils.read_images(os.path.join(images_dir,datatype[0])))
fake_images=np.array(utils.read_images(os.path.join(images_dir,datatype[1])))
images = np.concatenate((real_images, fake_images))
labels = np.array([0] * len(real_images) + [1] * len(fake_images))
# Resize and normalize images
ref_images, resize_images = [], []
for img in images:
    ori, norm = preprocess.resize_and_normalize(img, resize, normalize)
    ref_images.append(ori)
    resize_images.append(norm)

ref_images = np.array(ref_images)
resize_images = np.array(resize_images)

# Shuffle images and labels
indices = np.arange(resize_images.shape[0])
np.random.shuffle(indices)
ref_images = ref_images[indices]
resize_images = resize_images[indices]
labels = labels[indices]

# Apply texture extraction/edge detection techniques
if preprocess_method == "fft":
    processed_images = preprocess.apply_fft(resize_images)
    preprocess_method = f'{preprocess_method}'
elif preprocess_method == "lbp":
    radius = param_prepare['lbp']['radius']
    n_points = param_prepare['lbp']['n_points']
    method = param_prepare['lbp']['method']
    processed_images = preprocess.apply_lbp(resize_images, radius, n_points, method)
    preprocess_method = f'{preprocess_method}, radius:{radius}, n_points:{n_points}, method:{method}'
elif preprocess_method == "sobel":
    kernel = param_prepare['sobel']['kernel']
    processed_images = preprocess.apply_sobel(resize_images, kernel)
    preprocess_method = f'{preprocess_method}, kernel:{kernel}'
elif preprocess_method == "clahe":
    clip_limit = param_prepare['clahe']['clip_limit']
    tile_grid_size = tuple(param_prepare['clahe']['tile_grid_size'])
    processed_images = preprocess.apply_clahe(resize_images, clip_limit, tile_grid_size)
    preprocess_method = f'{preprocess_method}, clip_limit:{clip_limit} ,tile_grid_size:{tile_grid_size}'
elif preprocess_method == "none":
    processed_images = resize_images
    preprocess_method = f'{preprocess_method}'

processed_images = np.array(processed_images)

if extract_stats=="True":
    features = [preprocess.extract_statistics(processed_images) for img in processed_images]
    features = np.array(features.values())
else:
    features=processed_images

# Save preprocessed images and labels
utils.create_dir(output_dir)
np.save(os.path.join(output_dir, f'features_{prepare_benchmark}.npy'), features)
np.save(os.path.join(output_dir, f'labels_{prepare_benchmark}.npy'), labels)
print(f"Data has been successfully preprocessed and saved to {output_dir}")

# Calculate the total runtime
end_time = time.time()
runtime = end_time - start_time
print(f"Total runtime: {runtime:.2f} seconds")

log_data = {
    "dataset": data_dir,
    "prepare_benchmark": prepare_benchmark,
    "seed": seed,
    "train_test": train_test,
    "interpolation_resize": f"resize:{resize}, normalize:{normalize}, extract_stats:{extract_stats}",
    "preprocess_method": preprocess_method,
    "features_shape": features.shape,
    "labels_shape": labels.shape,
    "output_dir": output_dir,
        "runtime_seconds": runtime  # Log the runtime
}

# Path to log file
log_file_path = os.path.join(output_dir, "log.json")
utils.write_json(log_data, log_file_path)