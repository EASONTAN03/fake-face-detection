import yaml
import os 
import cv2
import numpy as np
import pandas as pd
import json
import shutil  # Import shutil to copy files
import random

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
seed = params['make_dataset']['seed']

param_prepare = params['prepare']
train_test = param_prepare['train_test']
resize = tuple(param_prepare['resize'])
color = param_prepare['color_mode']
normalize = param_prepare['normalize']
param_filter=param_prepare['filter']
filter = param_filter['method']
param_feature=param_prepare['feature_extract']
feature_extract = param_feature['method']

np.random.seed(seed)
random.seed(seed)

# Define input and output directories
data_dir = f'{dataset}_{benchmark}'
input_dir = os.path.join(interim_data_path, data_dir)
output_dir = os.path.join(processed_data_path, data_dir, train_test)
images_dir = os.path.join(input_dir, train_test)
datatype=["real","fake"]


real_images=np.array(utils.read_images(os.path.join(images_dir,datatype[0])))
fake_images=np.array(utils.read_images(os.path.join(images_dir,datatype[1])))
images=np.concatenate((real_images,fake_images))
labels=np.array([0]*len(real_images)+[1]*len(fake_images))

#Normalised images
# train_images = np.array([preprocess.resize_and_normalize(img, resize) for img in train_images], dtype=np.uint8)
# test_images = np.array([preprocess.resize_and_normalize(img, resize) for img in test_images], dtype=np.uint8)

ref_images = []
resize_images = []
for img in images:
    ori, norm = preprocess.resize_and_normalize(img, resize, normalize, color)
    ref_images.append(ori)
    resize_images.append(norm)
ref_images=np.array(ref_images)
resize_images=np.array(resize_images)

# Create an array of indices and shuffle them
indices = np.arange(resize_images.shape[0])
np.random.shuffle(indices) 
ref_images=ref_images[indices]
resize_images = resize_images[indices]
labels = labels[indices]

'''Apply filter/ edge detection techniques'''
if filter=="sobel":
    processed_images = preprocess.apply_sobel(resize_images)
    preprocess_method=f'{filter}'
elif filter=="gaussian":
    kernel=tuple(param_filter['gaussian']['kernel'])
    sigma=param_filter['gaussian']['sigma']
    processed_images = preprocess.apply_gaussian(resize_images, kernel, sigma)
    preprocess_method=f'{filter}, kernel:{kernel}, sigma:{sigma}'
elif filter=="gabor":
    sigma=param_filter['gabor']['sigma']
    frequency=param_filter['gabor']['frequency']
    processed_images = preprocess.apply_gabor(resize_images, sigma, frequency)
    preprocess_method=f'{filter}, sigma:{sigma}, frequency:{frequency}'
processed_images=np.array(processed_images)
print("Done filtering with output shape: ",processed_images.shape, np.min(processed_images), np.mean(processed_images), np.max(processed_images))
print(processed_images[0])

'''Apply feature extraction/reduction techniques'''
if feature_extract=="pca":
    components=param_feature['pca']['components']
    extracted_features = preprocess.apply_pca(processed_images, components)
    feature_extract_method=f'{feature_extract}, components:{components}'
elif feature_extract=="fft":
    extracted_features = preprocess.apply_fft(processed_images)
    feature_extract_method=f'{feature_extract}'
elif feature_extract=="sift":
    extracted_features = preprocess.apply_sift(processed_images)
    feature_extract_method=f'{feature_extract}'
elif feature_extract=="hog":
    extracted_features = preprocess.apply_hog(processed_images)
    feature_extract_method=f'{feature_extract}'
elif feature_extract=="ela":
    quality=param_feature['ela']['quality']
    extracted_features = preprocess.apply_ela(processed_images, quality)
    feature_extract_method=f'{feature_extract}, quality:{quality}'
elif feature_extract=="lbp":
    radius=param_feature['lbp']['radius']
    n_points=param_feature['lbp']['n_points']
    extracted_features = preprocess.apply_lbp(processed_images, radius, n_points)
    feature_extract_method=f'{feature_extract}, radius:{radius}, n_points:{n_points}'
elif feature_extract=="apply_landmark_detection":
    extracted_features = preprocess.apply_landmark_detection(processed_images)
    feature_extract_method=f'{feature_extract}'
extracted_features=np.array(extracted_features)
print("Done feature extraction with output shape: ",extracted_features.shape)

utils.plot_images(ref_images, processed_images, extracted_features, labels, num_images=6)

# Save preprocessed images and labels
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, f'{train_test}_features.npy'), extracted_features)
np.save(os.path.join(output_dir, f'{train_test}_labels.npy'), labels)
print(f"Data has been successfully preprocessed and saved to {output_dir}")

log_data = {
    "dataset": data_dir,
    "seed": seed,
    "train_test": train_test,
    "interpolation_resize": f"resize:{resize}, normalize:{normalize}, color_mode:{color}",
    "preprocess_method": preprocess_method,
    "feature_extract_method": feature_extract_method,
    "features_shape": extracted_features.shape,
    "labels_shape": labels.shape,
    "output_dir": output_dir
}

# Path to log file
log_file_path = os.path.join(processed_data_path, "log.json")
utils.write_json(log_data, log_file_path)