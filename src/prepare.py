import yaml
import os 
import cv2
import numpy as np
import pandas as pd
import json
import shutil  # Import shutil to copy files

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import operation_utils as utils
import preprocess_utils as preprocess
current_dir = os.getcwd()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

interim_data_path = config['dataset']['interim']
processed_data_path = config['dataset']['processed']
dataset = config['configs']['dataset']
benchmark = config['configs']['benchmark']

with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)
resize = tuple(params['prepare']['resize'])

# Define input and output directories
data_dir = f'{dataset}_{benchmark}'
input_dir = os.path.join(interim_data_path, data_dir)
output_dir = os.path.join(processed_data_path, data_dir)

# Paths to train and test directories
train_img_dir = os.path.join(input_dir, 'train')
test_img_dir = os.path.join(input_dir, 'test')
train_labels_path = os.path.join(input_dir, 'train_labels.npy')
test_labels_path = os.path.join(input_dir, 'test_labels.npy')
train_test_dir=['train','test']

#Input labels
train_images,test_images=utils.load_img(input_dir,train_test_dir)
train_labels = utils.load_labels(train_labels_path)
test_labels = utils.load_labels(test_labels_path)

# Preprocess images
train_images_preprocessed = np.array([preprocess.resize_and_normalize(img, resize) for img in train_images] ,type=np.UNIT8)
test_images_preprocessed = np.array([preprocess.resize_and_normalize(img, resize) for img in test_images] ,type=np.UNIT8)

# Save preprocessed images and labels
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, 'train_images.npy'), train_images_preprocessed)
np.save(os.path.join(output_dir, 'test_images.npy'), test_images_preprocessed)
np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)

print(f"Data has been successfully preprocessed and saved to {output_dir}")

log_data = {
    "total_train_images": len(train_images),
    "total_test_images": len(test_images),
    "resize": resize,
    "preprocess_method": 'resize',
    "color_space": 'rgb',
    "datatype": str(train_images_preprocessed.dtype)
}

# Path to log file
log_file_path = os.path.join(output_dir, "log.json")
utils.write_json(log_data, log_file_path)